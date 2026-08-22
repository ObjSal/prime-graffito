//! Post-quantum sealing extension for directed PRIVATE single-recipient
//! notes — TWO optional, composable extra key layers, hybrid ON TOP of the
//! existing dm.rs static-static ECDH (never a replacement for it):
//!
//! - **ML-KEM** (`FLAG_MLKEM`, envelope.rs): a FIPS-203 key-encapsulation
//!   ciphertext addressed to the recipient's ML-KEM encapsulation key.
//!   Post-quantum forward secrecy for the note — but ONLY the recipient can
//!   ever decapsulate it. The sender holds no ML-KEM secret at all, so a
//!   sender re-reading their own sent note (`unlock_sent`) can NEVER recover
//!   a KEM-layered note — see [`Error::SenderCannotReopen`]. This is by
//!   design, not a bug: it is the same property that makes the KEM useful
//!   (an attacker who later breaks ECDH via a quantum computer still can't
//!   read the note without the recipient's KEM secret, which never touches
//!   the chain).
//! - **Password** (`FLAG_PW`, envelope.rs): an Argon2id-stretched shared
//!   password, layered in alongside the ECDH secret. Recoverable by EITHER
//!   party who knows the password (including sender re-read), since it adds
//!   no asymmetric secret.
//!
//! Both are valid only on a DIRECTED PRIVATE SINGLE-recipient note (FLAG_MULTI
//! is incompatible — see envelope.rs's validity rule) and can be combined.
//!
//! ## Wire format
//!
//! The note BODY (after the ASCII header, PLAN-pnte-redesign.md) gains
//! prefix blocks ahead of today's sealed blob, in this exact order:
//!
//! ```text
//! [alg_id(1) || mlkem_ct(ct_len(alg))]   iff FLAG_MLKEM
//! [salt(16) || t(1) || m_log2(1) || p(1)] iff FLAG_PW      (19 bytes)
//! nonce(24) || ciphertext || tag(16)                        (crypt.rs, unchanged)
//! ```
//!
//! AAD is UNCHANGED from dm.rs's `dm_aad(sender_x, recipient_x, outpoint)` —
//! tampering the KEM ciphertext is caught via ML-KEM's implicit rejection
//! (a bad ct decapsulates to a pseudorandom-but-wrong shared secret, never
//! an error) flowing into the WRONG sealing key, which then fails the AEAD
//! tag check — never a distinct error path, exactly like a wrong password.
//!
//! ## Key derivation (NEW domain — dm.rs's v1 salts/keys are untouched)
//!
//! ```text
//! sealing_key = HKDF-SHA256(
//!     salt = "prime-graffito/dm-pq/v1",
//!     ikm  = ecdh_shared_x(32) || [mlkem_ss(32) if FLAG_MLKEM] || [pw_key(32) if FLAG_PW],
//! ).expand(info = "dm-enc-pq/v1" || pq_flags_byte || [alg_id if FLAG_MLKEM], 32)
//! ```
//!
//! `ecdh_shared_x` is the SAME `dm::ecdh_shared_x` used by v1 — this module
//! never reimplements or alters it. `pq_flags_byte = flags & 0x30` (just the
//! two pq bits).
//!
//! ## RNG rule (CRITICAL — read before touching this file)
//!
//! This crate runs on a device where ALL entropy must route through
//! `getrandom` 0.2 (the workspace's vendored TRNG `[patch]` override —
//! `RANDOMNESS-AUDIT-2026-08-01.md`). This module MUST NOT use `rand`,
//! `rand_core::OsRng`, or any API that could reach getrandom 0.3/0.4. Every
//! random byte here is drawn by calling `getrandom::getrandom` directly and
//! handed to a DETERMINISTIC ml-kem API (`generate_deterministic`,
//! `encapsulate_deterministic`, both gated behind ml-kem's `deterministic`
//! Cargo feature) — the plain `generate`/`encapsulate` methods (which
//! demand a `CryptoRngCore`) are never called anywhere in this file.
//! `cargo tree -p notes-core -i getrandom@0.3` must stay empty.

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::{Zeroize, Zeroizing};

use ml_kem::array::Array;
use ml_kem::kem::{Decapsulate, EncapsulationKey};
use ml_kem::{
    B32, Ciphertext, EncapsulateDeterministic, Encoded, EncodedSizeUser, KemCore,
    MlKem1024, MlKem512, MlKem768,
};

use crate::envelope::{FLAG_MLKEM, FLAG_PW};
use crate::{crypt, dm, Error};

// ---------------------------------------------------------------------
// ML-KEM parameter sets
// ---------------------------------------------------------------------

/// Which FIPS-203 ML-KEM parameter set. notes-core is level-agnostic — all
/// three are equally "supported"; 768 is the recommended default for new
/// callers, encoded as a plain doc recommendation, not an enforced default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlKemAlg {
    MlKem512,
    MlKem768,
    MlKem1024,
}

impl MlKemAlg {
    pub fn id(self) -> u8 {
        match self {
            MlKemAlg::MlKem512 => 0x01,
            MlKemAlg::MlKem768 => 0x02,
            MlKemAlg::MlKem1024 => 0x03,
        }
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0x01 => Some(MlKemAlg::MlKem512),
            0x02 => Some(MlKemAlg::MlKem768),
            0x03 => Some(MlKemAlg::MlKem1024),
            _ => None,
        }
    }

    /// Encapsulation key length, bytes (FIPS 203).
    pub fn ek_len(self) -> usize {
        match self {
            MlKemAlg::MlKem512 => 800,
            MlKemAlg::MlKem768 => 1184,
            MlKemAlg::MlKem1024 => 1568,
        }
    }

    /// Ciphertext length, bytes (FIPS 203).
    pub fn ct_len(self) -> usize {
        match self {
            MlKemAlg::MlKem512 => 768,
            MlKemAlg::MlKem768 => 1088,
            MlKemAlg::MlKem1024 => 1568,
        }
    }

    /// Serialized (expanded) decapsulation-key length, bytes (FIPS 203) —
    /// the length [`MlKemSecret::Expanded`] must have for this alg.
    pub fn dk_len(self) -> usize {
        match self {
            MlKemAlg::MlKem512 => 1632,
            MlKemAlg::MlKem768 => 2400,
            MlKemAlg::MlKem1024 => 3168,
        }
    }
}

/// Dispatch a block of code generic over ml-kem's concrete `Kem<P>` type for
/// the given [`MlKemAlg`] — the three parameter sets are distinct Rust
/// types (no runtime polymorphism in the crate), so this macro is the
/// dispatch point every alg-generic operation below goes through.
macro_rules! with_alg {
    ($alg:expr, $K:ident => $body:block) => {
        match $alg {
            MlKemAlg::MlKem512 => {
                type $K = MlKem512;
                $body
            }
            MlKemAlg::MlKem768 => {
                type $K = MlKem768;
                $body
            }
            MlKemAlg::MlKem1024 => {
                type $K = MlKem1024;
                $body
            }
        }
    };
}

fn b32_from(bytes: [u8; 32]) -> B32 {
    Array::from(bytes)
}

/// A (decapsulation key, encapsulation key) pair generated from a 64-byte
/// `(d, z)` seed (FIPS 203's key-generation randomness) — the seed ALONE
/// reconstructs the pair, so it is the only thing that needs to be stored
/// or exported (see the armor format below).
#[derive(Clone)]
pub struct MlKemKeypair {
    alg: MlKemAlg,
    ek: Vec<u8>,
    seed: [u8; 64],
}

impl Drop for MlKemKeypair {
    fn drop(&mut self) {
        self.seed.zeroize();
    }
}

impl MlKemKeypair {
    /// Fresh keypair: draws 64 bytes from the TRNG (`getrandom`, per the
    /// module's RNG rule) and derives deterministically from them.
    pub fn generate(alg: MlKemAlg) -> Result<Self, Error> {
        let mut seed = [0u8; 64];
        getrandom::getrandom(&mut seed).map_err(|_| Error::Entropy)?;
        Ok(Self::from_seed(alg, &seed))
    }

    /// Deterministic (re-)derivation from an existing 64-byte seed — no
    /// entropy draw, infallible. `seed = d(32) || z(32)` (FIPS 203).
    pub fn from_seed(alg: MlKemAlg, seed: &[u8; 64]) -> Self {
        let mut d = [0u8; 32];
        let mut z = [0u8; 32];
        d.copy_from_slice(&seed[..32]);
        z.copy_from_slice(&seed[32..]);
        let ek = with_alg!(alg, K => {
            let (_dk, ek) = <K as KemCore>::generate_deterministic(&b32_from(d), &b32_from(z));
            ek.as_bytes().as_slice().to_vec()
        });
        MlKemKeypair { alg, ek, seed: *seed }
    }

    pub fn alg(&self) -> MlKemAlg {
        self.alg
    }

    /// The encapsulation key (public half) — [`MlKemAlg::ek_len`] bytes.
    pub fn ek(&self) -> &[u8] {
        &self.ek
    }

    pub fn seed(&self) -> &[u8; 64] {
        &self.seed
    }

    /// The decapsulation secret in its cheapest form (the seed — see
    /// [`MlKemSecret`]).
    pub fn secret(&self) -> MlKemSecret {
        MlKemSecret::Seed(self.seed)
    }

    /// The decapsulation secret in its EXPANDED (serialized decapsulation
    /// key) form — [`MlKemAlg::dk_len`] bytes. Same key as [`Self::secret`],
    /// just pre-derived rather than reconstructed from the seed on demand;
    /// this is the form a key imported from elsewhere (e.g. OpenPGP) may
    /// arrive in when only the expanded key, not the original seed, is
    /// available.
    pub fn expanded_secret(&self) -> MlKemSecret {
        let mut d = [0u8; 32];
        let mut z = [0u8; 32];
        d.copy_from_slice(&self.seed[..32]);
        z.copy_from_slice(&self.seed[32..]);
        let bytes = with_alg!(self.alg, K => {
            let (dk, _ek) = <K as KemCore>::generate_deterministic(&b32_from(d), &b32_from(z));
            dk.as_bytes().as_slice().to_vec()
        });
        MlKemSecret::Expanded(bytes)
    }

    pub fn fingerprint(&self) -> String {
        fingerprint(self.alg, &self.ek)
    }
}

/// A decapsulation secret in either of the two forms `unlock_received`
/// accepts: the 64-byte seed (reconstructs the full keypair deterministically
/// — the normal on-device case), or the pre-expanded serialized decapsulation
/// key bytes (needed for keys imported from elsewhere, e.g. OpenPGP, where
/// only the expanded form may be available). Zeroized on drop.
pub enum MlKemSecret {
    Seed([u8; 64]),
    Expanded(Vec<u8>),
}

impl Drop for MlKemSecret {
    fn drop(&mut self) {
        match self {
            MlKemSecret::Seed(s) => s.zeroize(),
            MlKemSecret::Expanded(v) => v.zeroize(),
        }
    }
}

impl Clone for MlKemSecret {
    fn clone(&self) -> Self {
        match self {
            MlKemSecret::Seed(s) => MlKemSecret::Seed(*s),
            MlKemSecret::Expanded(v) => MlKemSecret::Expanded(v.clone()),
        }
    }
}

/// SHA256(alg_id_byte || ek), first 16 lowercase hex chars, grouped in
/// fours: `"xxxx xxxx xxxx xxxx"`.
pub fn fingerprint(alg: MlKemAlg, ek: &[u8]) -> String {
    use sha2::Digest;
    let mut data = Vec::with_capacity(1 + ek.len());
    data.push(alg.id());
    data.extend_from_slice(ek);
    let digest = Sha256::digest(&data);
    let hex_str = hex::encode(&digest[..8]);
    format!("{} {} {} {}", &hex_str[0..4], &hex_str[4..8], &hex_str[8..12], &hex_str[12..16])
}

fn ek_encoded<K: KemCore>(bytes: &[u8]) -> Result<Encoded<K::EncapsulationKey>, Error> {
    Array::try_from(bytes).map_err(|_| Error::MlKemAlgMismatch)
}

fn dk_encoded<K: KemCore>(bytes: &[u8]) -> Result<Encoded<K::DecapsulationKey>, Error> {
    Array::try_from(bytes).map_err(|_| Error::MlKemAlgMismatch)
}

/// Encapsulate a fresh shared secret to `ek` (the recipient's encapsulation
/// key bytes). Randomness (the FIPS-203 `m` input) is drawn via `getrandom`
/// and handed to the deterministic API — see the module's RNG rule.
fn encapsulate(alg: MlKemAlg, ek_bytes: &[u8]) -> Result<(Vec<u8>, [u8; 32]), Error> {
    if ek_bytes.len() != alg.ek_len() {
        return Err(Error::MlKemAlgMismatch);
    }
    let mut m = [0u8; 32];
    getrandom::getrandom(&mut m).map_err(|_| Error::Entropy)?;
    let m = b32_from(m);
    with_alg!(alg, K => {
        let enc = ek_encoded::<K>(ek_bytes)?;
        let ek: EncapsulationKey<_> = <K as KemCore>::EncapsulationKey::from_bytes(&enc);
        let (ct, ss): (Ciphertext<K>, _) =
            ek.encapsulate_deterministic(&m).map_err(|_| Error::DecryptFailed)?;
        let mut ss_out = [0u8; 32];
        ss_out.copy_from_slice(ss.as_slice());
        Ok((ct.as_slice().to_vec(), ss_out))
    })
}

/// Decapsulate `ct` under `secret`. ML-KEM decapsulation is INFALLIBLE by
/// construction (implicit rejection: a wrong key/tampered ct silently
/// yields a pseudorandom-but-wrong shared secret rather than an error) —
/// the only errors this returns are STRUCTURAL (wrong ct length for `alg`,
/// or an `Expanded` secret whose length doesn't match `alg`'s decapsulation
/// key size), never "wrong key" — that surfaces later as an AEAD failure.
fn decapsulate(alg: MlKemAlg, secret: &MlKemSecret, ct: &[u8]) -> Result<[u8; 32], Error> {
    if ct.len() != alg.ct_len() {
        return Err(Error::MlKemAlgMismatch);
    }
    if let MlKemSecret::Expanded(bytes) = secret {
        if bytes.len() != alg.dk_len() {
            return Err(Error::MlKemAlgMismatch);
        }
    }
    with_alg!(alg, K => {
        let dk: <K as KemCore>::DecapsulationKey = match secret {
            MlKemSecret::Seed(seed) => {
                let mut d = [0u8; 32];
                let mut z = [0u8; 32];
                d.copy_from_slice(&seed[..32]);
                z.copy_from_slice(&seed[32..]);
                let (dk, _ek) = <K as KemCore>::generate_deterministic(&b32_from(d), &b32_from(z));
                dk
            }
            MlKemSecret::Expanded(bytes) => {
                let enc = dk_encoded::<K>(bytes)?;
                <K as KemCore>::DecapsulationKey::from_bytes(&enc)
            }
        };
        let ct_arr: Ciphertext<K> = Array::try_from(ct).map_err(|_| Error::MlKemAlgMismatch)?;
        let ss = dk.decapsulate(&ct_arr).map_err(|_| Error::DecryptFailed)?;
        let mut out = [0u8; 32];
        out.copy_from_slice(ss.as_slice());
        Ok(out)
    })
}

// ---------------------------------------------------------------------
// Armor (import/export) — format shared with a sibling implementation;
// do not deviate.
// ---------------------------------------------------------------------

const PRIVATE_LABEL: &str = "GRAFFITO ML-KEM PRIVATE KEY";
const PUBLIC_LABEL: &str = "GRAFFITO ML-KEM PUBLIC KEY";
/// Leading byte of every armored payload: format version, NOT the ML-KEM
/// alg id (that's the second byte).
const ARMOR_VERSION: u8 = 0x01;
const ARMOR_LINE_LEN: usize = 64;

fn armor_wrap(label: &str, payload: &[u8]) -> String {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
    let mut out = String::with_capacity(b64.len() + b64.len() / ARMOR_LINE_LEN + 64);
    out.push_str("-----BEGIN ");
    out.push_str(label);
    out.push_str("-----\n");
    for chunk in b64.as_bytes().chunks(ARMOR_LINE_LEN) {
        out.push_str(std::str::from_utf8(chunk).expect("base64 is ascii"));
        out.push('\n');
    }
    out.push_str("-----END ");
    out.push_str(label);
    out.push_str("-----\n");
    out
}

/// Liberal about whitespace/line length inside the body, strict about the
/// BEGIN/END magic labels: returns the decoded payload bytes.
fn armor_unwrap(label: &str, armored: &str) -> Result<Vec<u8>, Error> {
    use base64::Engine as _;
    let begin = format!("-----BEGIN {label}-----");
    let end = format!("-----END {label}-----");
    let start = armored.find(&begin).ok_or(Error::Decode("pq armor: missing BEGIN label"))?;
    let after_begin = start + begin.len();
    let end_pos =
        armored[after_begin..].find(&end).ok_or(Error::Decode("pq armor: missing END label"))?;
    let body = &armored[after_begin..after_begin + end_pos];
    let b64: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|_| Error::Decode("pq armor: bad base64"))
}

/// Export a keypair's PRIVATE (seed) form: `0x01 || alg_id(1) || seed(64)`.
pub fn export_private(alg: MlKemAlg, seed: &[u8; 64]) -> String {
    let mut payload = Vec::with_capacity(2 + 64);
    payload.push(ARMOR_VERSION);
    payload.push(alg.id());
    payload.extend_from_slice(seed);
    armor_wrap(PRIVATE_LABEL, &payload)
}

/// Import an armored private key, returning `(alg, seed)`.
pub fn import_private(armored: &str) -> Result<(MlKemAlg, [u8; 64]), Error> {
    let payload = armor_unwrap(PRIVATE_LABEL, armored)?;
    if payload.len() != 2 + 64 {
        return Err(Error::Decode("pq armor: bad private-key payload length"));
    }
    if payload[0] != ARMOR_VERSION {
        return Err(Error::Decode("pq armor: unsupported version byte"));
    }
    let alg = MlKemAlg::from_id(payload[1]).ok_or(Error::Decode("pq armor: unknown alg id"))?;
    let mut seed = [0u8; 64];
    seed.copy_from_slice(&payload[2..]);
    Ok((alg, seed))
}

/// Export a keypair's PUBLIC (ek) form: `0x01 || alg_id(1) || ek`.
pub fn export_public(alg: MlKemAlg, ek: &[u8]) -> String {
    let mut payload = Vec::with_capacity(2 + ek.len());
    payload.push(ARMOR_VERSION);
    payload.push(alg.id());
    payload.extend_from_slice(ek);
    armor_wrap(PUBLIC_LABEL, &payload)
}

/// Import an armored public key, returning `(alg, ek)`.
pub fn import_public(armored: &str) -> Result<(MlKemAlg, Vec<u8>), Error> {
    let payload = armor_unwrap(PUBLIC_LABEL, armored)?;
    if payload.len() < 2 {
        return Err(Error::Decode("pq armor: truncated public-key payload"));
    }
    if payload[0] != ARMOR_VERSION {
        return Err(Error::Decode("pq armor: unsupported version byte"));
    }
    let alg = MlKemAlg::from_id(payload[1]).ok_or(Error::Decode("pq armor: unknown alg id"))?;
    let ek = payload[2..].to_vec();
    if ek.len() != alg.ek_len() {
        return Err(Error::Decode("pq armor: wrong ek length for alg"));
    }
    Ok((alg, ek))
}

// ---------------------------------------------------------------------
// Seed-derived receive keypair — cross-app recovery contract, FROZEN
// once shipped
// ---------------------------------------------------------------------
//
// Per-notebook seed-derived ML-KEM receive keys, relocated here from the
// Mac app's `app-core/src/pqkeys.rs` (same precedent as `keys::
// enc_key_from_leaf`'s relocation) so the Passport Prime device app and
// the Mac app derive BYTE-IDENTICAL keys from the same notebook leaf
// secret — a notebook's leaf secret fully determines its ML-KEM receive
// keys at every level, on device and Mac alike, so recovery from the
// seed words alone reproduces exactly the keys a contact already has
// stored, on either platform.
//
// ```text
// seed64 = HKDF-SHA256(salt = "graffito/mlkem/v1", ikm = leaf_secret)
//            .expand(info = "seed/v1" || alg_id(1), 64)
// MlKemKeypair::from_seed(alg, &seed64)
// ```
//
// `alg_id` (`MlKemAlg::id()`) folds into the HKDF `info`, making the
// three levels independent draws from the same `leaf_secret` — never the
// same seed truncated/reused three ways, so compromising one level's
// decapsulation key reveals nothing about the other two.
//
// FROZEN FOREVER once shipped: receive keys derived here are advertised
// to peers (a contact stores the resulting `ek`), so changing the salt,
// info prefix, alg-id placement, or `MlKemKeypair::from_seed`'s (d, z)
// byte layout orphans every already-shared pq receive key. Pinned by
// `pinned_derivation_vectors_per_level` below — treat a failure there as
// SHIP-BLOCKING, never "fix" it by updating the hex.

const MLKEM_SEED_SALT: &[u8] = b"graffito/mlkem/v1";
const MLKEM_SEED_INFO_PREFIX: &[u8] = b"seed/v1";

/// The HKDF step alone (seed derivation, no ML-KEM keygen) — broken out
/// so [`mlkem_keypair_from_leaf`] and any future caller that only needs
/// the seed (never the expensive keygen) share one implementation.
/// Zeroizing: the 64-byte seed is exactly as sensitive as the
/// decapsulation key it deterministically reconstructs. FROZEN — see the
/// section doc above.
pub fn mlkem_seed_from_leaf(leaf_secret: &[u8; 32], alg: MlKemAlg) -> Zeroizing<[u8; 64]> {
    let hk = Hkdf::<Sha256>::new(Some(MLKEM_SEED_SALT), leaf_secret);
    let mut info = Vec::with_capacity(MLKEM_SEED_INFO_PREFIX.len() + 1);
    info.extend_from_slice(MLKEM_SEED_INFO_PREFIX);
    info.push(alg.id());
    let mut okm = Zeroizing::new([0u8; 64]);
    hk.expand(&info, &mut *okm).expect("64 bytes is a valid HKDF length");
    okm
}

/// Deterministically derive a notebook's ML-KEM receive keypair at `alg`
/// from its `leaf_secret` — see the section doc above for the exact
/// derivation. Infallible (HKDF expand to a fixed 64-byte length never
/// fails, and [`MlKemKeypair::from_seed`] is deterministic keygen, no
/// entropy draw).
pub fn mlkem_keypair_from_leaf(leaf_secret: &[u8; 32], alg: MlKemAlg) -> MlKemKeypair {
    let seed = mlkem_seed_from_leaf(leaf_secret, alg);
    MlKemKeypair::from_seed(alg, &seed)
}

// ---------------------------------------------------------------------
// Argon2id password layer
// ---------------------------------------------------------------------

/// Production Argon2id parameters: t=3 iterations, m=2^15 KiB (32 MiB),
/// p=1 lane.
///
/// `m` was 2^16 (64 MiB, Sal's original brief) until the 2026-08-22 crypto
/// audit (F2, `private/SECURITY-AUDIT-CRYPTO-2026-08-22.md`): a Passport
/// Prime has ~59-60 MB of free heap, so a single 64 MiB Argon2 allocation
/// plausibly fails on hardware — and both apps must EMIT one value the
/// weakest platform can also UNLOCK, or Mac-sealed FLAG_PW notes become
/// unreadable on device. 32 MiB at t=3 still clears RFC 9106's and OWASP's
/// recommended Argon2id floors. Notes already sealed at m=2^16 remain
/// decodable everywhere the memory exists ([`validate_pw_params`] accepts
/// up to 16) — the wire prefix self-describes its params, so this constant
/// only shapes NEW notes and is safe to lower again (never raise past the
/// decode cap without bumping the cap a release earlier).
pub const PW_PROD_T: u32 = 3;
pub const PW_PROD_M_LOG2: u8 = 15;
pub const PW_PROD_P: u32 = 1;

/// Decode-side cap on a received note's `m_log2`. These params are
/// ATTACKER-CONTROLLED on-chain bytes, and Argon2 allocates `2^m_log2` KiB
/// up front — before the 2026-08-22 audit (F1) this cap was 24, letting a
/// hostile note demand a 16 GiB allocation at unlock time (an instant
/// process abort on a 128 MB device, an OOM grind elsewhere). 16 is the
/// largest value any honest emitter has EVER written ([`PW_PROD_M_LOG2`]
/// was 16 from the feature's ship date until the same audit lowered it to
/// 15), so capping here rejects nothing legitimate. If production params
/// are ever raised past 16, ship the cap bump one release BEFORE the
/// emitter bump, or old builds will reject the new notes.
pub const PW_MAX_M_LOG2: u8 = 16;
/// Decode-side cap on a received note's `t` (pass count). Honest emitters
/// have only ever written [`PW_PROD_T`] = 3; 16 leaves generous headroom
/// while bounding the CPU an attacker can demand (255 passes over a
/// 64 MiB arena ≈ minutes of compute per unlock attempt). Same
/// raise-the-cap-first rule as [`PW_MAX_M_LOG2`].
pub const PW_MAX_T: u32 = 16;

/// Bounds on parsed (untrusted, on-chain) Argon2 params — guards
/// [`pw_key`] against a malformed/hostile PW prefix block. `m_log2` and
/// `t` are capped at [`PW_MAX_M_LOG2`]/[`PW_MAX_T`] (see those consts for
/// the threat model); `t`/`p` need to be >= 1 (Argon2's own minimums).
fn validate_pw_params(t: u32, m_log2: u8, p: u32) -> Result<(), Error> {
    if t < 1 || p < 1 || p > 0x00ff_ffff {
        return Err(Error::Decode("pq: invalid argon2 params"));
    }
    if t > PW_MAX_T {
        return Err(Error::Decode("pq: argon2 pass count too large"));
    }
    if m_log2 > PW_MAX_M_LOG2 {
        return Err(Error::Decode("pq: argon2 memory cost too large"));
    }
    let m_cost = 1u32 << m_log2;
    if m_cost < 8 * p {
        return Err(Error::Decode("pq: argon2 memory cost too small for lane count"));
    }
    Ok(())
}

/// Argon2id(v0x13) key derivation: `m = 2^m_log2` KiB, `t` iterations, `p`
/// lanes, 32-byte output. `(t, m_log2, p)` must already be valid Argon2
/// parameters (production callers use [`PW_PROD_T`] / [`PW_PROD_M_LOG2`] /
/// [`PW_PROD_P`], which always are; callers parsing `(t, m_log2, p)` off
/// the chain MUST run [`validate_pw_params`] first —
/// `unlock_received`/`unlock_sent` do this before ever calling `pw_key`).
/// Invalid params return `Error::Decode` (never reachable from this
/// module's own decode paths).
///
/// The Argon2 memory arena is allocated FALLIBLY (`try_reserve_exact` +
/// `hash_password_into_with_memory`) and a failure returns
/// [`Error::OutOfMemory`] — 2026-08-22 audit F2: on a Passport Prime
/// (~59-60 MB free heap) the infallible-allocation path
/// (`hash_password_into`, which `vec![...]`s the arena) turns "not enough
/// memory for these params" into an uncatchable process ABORT. Same
/// blocks, same algorithm, byte-identical output — pinned by
/// `pw_key_vectors_are_pinned` in tests/pq.rs.
pub fn pw_key(
    password: &str,
    salt: &[u8; 16],
    t: u32,
    m_log2: u8,
    p: u32,
) -> Result<[u8; 32], Error> {
    let m_cost = 1u32 << m_log2;
    let params = argon2::Params::new(m_cost, t, p, Some(32))
        .map_err(|_| Error::Decode("pq: invalid argon2 params"))?;
    let block_count = params.block_count();
    let argon2 = argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut blocks: Vec<argon2::Block> = Vec::new();
    blocks.try_reserve_exact(block_count).map_err(|_| Error::OutOfMemory)?;
    blocks.resize(block_count, argon2::Block::default());
    let mut out = [0u8; 32];
    argon2
        .hash_password_into_with_memory(password.as_bytes(), salt, &mut out, &mut blocks)
        .map_err(|_| Error::Decode("pq: invalid argon2 params"))?;
    Ok(out)
}

// ---------------------------------------------------------------------
// HKDF key derivation (NEW domain, dm-pq/v1 — see module doc)
// ---------------------------------------------------------------------

const DM_PQ_SALT: &[u8] = b"prime-graffito/dm-pq/v1";
const DM_PQ_INFO_PREFIX: &[u8] = b"dm-enc-pq/v1";

/// `pq_flags` here is JUST the two pq bits (`flags & 0x30`), matching the
/// module doc's `pq_flags_byte`.
fn derive_pq_key(
    pq_flags: u8,
    alg: Option<MlKemAlg>,
    shared_x: &[u8; 32],
    mlkem_ss: Option<&[u8; 32]>,
    pw_key_bytes: Option<&[u8; 32]>,
) -> [u8; 32] {
    let mut ikm = Vec::with_capacity(96);
    ikm.extend_from_slice(shared_x);
    if let Some(ss) = mlkem_ss {
        ikm.extend_from_slice(ss);
    }
    if let Some(pw) = pw_key_bytes {
        ikm.extend_from_slice(pw);
    }
    let mut info = Vec::with_capacity(DM_PQ_INFO_PREFIX.len() + 2);
    info.extend_from_slice(DM_PQ_INFO_PREFIX);
    info.push(pq_flags & (FLAG_PW | FLAG_MLKEM));
    if let Some(a) = alg {
        info.push(a.id());
    }
    let hk = Hkdf::<Sha256>::new(Some(DM_PQ_SALT), &ikm);
    let mut okm = [0u8; 32];
    hk.expand(&info, &mut okm).expect("32 bytes is a valid HKDF length");
    okm
}

// ---------------------------------------------------------------------
// Seal / unlock
// ---------------------------------------------------------------------

/// Which extra layer(s) to seal a directed-private note under. At least one
/// field must be `Some` — callers with neither use the plain v1 `dm.rs`
/// path instead (see `bundle::compose_directed_note_with_change_amount`).
#[derive(Clone, Copy)]
pub struct SealLayers<'a> {
    pub mlkem_ek: Option<(MlKemAlg, &'a [u8])>,
    pub password: Option<&'a str>,
}

impl<'a> SealLayers<'a> {
    /// The envelope pq flag bits (`FLAG_MLKEM`/`FLAG_PW`) this set of
    /// layers would produce — usable by callers (e.g. the compose cost
    /// estimator) before any crypto runs.
    pub fn flags(&self) -> u8 {
        (if self.mlkem_ek.is_some() { FLAG_MLKEM } else { 0 })
            | (if self.password.is_some() { FLAG_PW } else { 0 })
    }
}

/// Prefix-byte overhead [`seal_directed_pq`] adds on top of
/// `crypt::SEAL_OVERHEAD` for the given pq flags/alg — pure arithmetic, for
/// the compose cost estimator (mirrors `envelope::payload_lens_for`'s
/// no-crypto-needed contract). `alg` is required when `FLAG_MLKEM` is set
/// (the ciphertext length depends on it); ignored otherwise.
pub fn pq_overhead(pq_flags: u8, alg: Option<MlKemAlg>) -> usize {
    let mut n = 0;
    if pq_flags & FLAG_MLKEM != 0 {
        n += 1 + alg.map_or(0, MlKemAlg::ct_len);
    }
    if pq_flags & FLAG_PW != 0 {
        n += 19; // salt(16) || t(1) || m_log2(1) || p(1)
    }
    n
}

/// Sender side: seal a directed-private body under one or both pq layers,
/// hybrid over the same static-static ECDH `dm.rs` uses for v1 (never
/// replacing it). Returns `(pq_flags, full_body)` — `full_body` is the
/// prefix block(s) followed by the ordinary sealed blob (see the module
/// doc's wire format); the caller envelopes it exactly like any other
/// private directed body (`FLAG_PRIVATE | FLAG_DIRECTED | pq_flags`).
pub fn seal_directed_pq(
    my_tweaked_seckey: &[u8; 32],
    my_output_x: &[u8; 32],
    recipient_x: &[u8; 32],
    outpoint: &[u8; 36],
    plaintext: &[u8],
    layers: SealLayers,
) -> Result<(u8, Vec<u8>), Error> {
    let pq_flags = layers.flags();
    if pq_flags == 0 {
        return Err(Error::Envelope("pq: at least one seal layer required"));
    }

    let mut prefix = Vec::new();
    let mut mlkem_ss: Option<[u8; 32]> = None;
    let mut alg_used: Option<MlKemAlg> = None;
    if let Some((alg, ek)) = layers.mlkem_ek {
        let (ct, ss) = encapsulate(alg, ek)?;
        prefix.push(alg.id());
        prefix.extend_from_slice(&ct);
        mlkem_ss = Some(ss);
        alg_used = Some(alg);
    }

    let mut pw_key_bytes: Option<[u8; 32]> = None;
    if let Some(password) = layers.password {
        let mut salt = [0u8; 16];
        getrandom::getrandom(&mut salt).map_err(|_| Error::Entropy)?;
        let key = pw_key(password, &salt, PW_PROD_T, PW_PROD_M_LOG2, PW_PROD_P)?;
        prefix.extend_from_slice(&salt);
        prefix.push(PW_PROD_T as u8);
        prefix.push(PW_PROD_M_LOG2);
        prefix.push(PW_PROD_P as u8);
        pw_key_bytes = Some(key);
    }

    let shared_x = dm::ecdh_shared_x(my_tweaked_seckey, recipient_x)?;
    let key = derive_pq_key(pq_flags, alg_used, &shared_x, mlkem_ss.as_ref(), pw_key_bytes.as_ref());
    let aad = dm::dm_aad(my_output_x, recipient_x, outpoint);
    let sealed = crypt::seal_aad(&key, &aad, plaintext)?;

    let mut full_body = prefix;
    full_body.extend_from_slice(&sealed);
    Ok((pq_flags, full_body))
}

struct MlKemPrefix<'a> {
    alg: MlKemAlg,
    ct: &'a [u8],
}

fn parse_mlkem_prefix(body: &[u8]) -> Result<(MlKemPrefix<'_>, &[u8]), Error> {
    let alg_id = *body.first().ok_or(Error::Decode("pq: truncated mlkem prefix"))?;
    let alg = MlKemAlg::from_id(alg_id).ok_or(Error::Decode("pq: unknown mlkem alg id"))?;
    let ct_len = alg.ct_len();
    if body.len() < 1 + ct_len {
        return Err(Error::Decode("pq: truncated mlkem ciphertext"));
    }
    Ok((MlKemPrefix { alg, ct: &body[1..1 + ct_len] }, &body[1 + ct_len..]))
}

struct PwPrefix {
    salt: [u8; 16],
    t: u32,
    m_log2: u8,
    p: u32,
}

fn parse_pw_prefix(body: &[u8]) -> Result<(PwPrefix, &[u8]), Error> {
    if body.len() < 19 {
        return Err(Error::Decode("pq: truncated pw params"));
    }
    let mut salt = [0u8; 16];
    salt.copy_from_slice(&body[..16]);
    let t = body[16] as u32;
    let m_log2 = body[17];
    let p = body[18] as u32;
    validate_pw_params(t, m_log2, p)?;
    Ok((PwPrefix { salt, t, m_log2, p }, &body[19..]))
}

/// A note recovered from chain data whose pq layer(s) have NOT been
/// unlocked yet — everything needed to attempt [`unlock_received`] or
/// [`unlock_sent`] once the caller supplies the missing secret(s). See
/// `bundle::RecoveredNote::locked`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockedBody {
    pub pq_flags: u8,
    pub body: Vec<u8>,
    pub sender_x: [u8; 32],
    pub recipient_x: [u8; 32],
    pub outpoint: [u8; 36],
}

mod locked_body_serde {
    use super::LockedBody;
    use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Wire {
        pq_flags: u8,
        body: String,
        sender_x: String,
        recipient_x: String,
        outpoint: String,
    }

    fn hex_arr<const N: usize>(s: &str) -> Result<[u8; N], String> {
        let bytes = hex::decode(s).map_err(|e| e.to_string())?;
        <[u8; N]>::try_from(bytes.as_slice()).map_err(|_| format!("expected {N} bytes"))
    }

    impl Serialize for LockedBody {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            Wire {
                pq_flags: self.pq_flags,
                body: hex::encode(&self.body),
                sender_x: hex::encode(self.sender_x),
                recipient_x: hex::encode(self.recipient_x),
                outpoint: hex::encode(self.outpoint),
            }
            .serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for LockedBody {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let w = Wire::deserialize(d)?;
            Ok(LockedBody {
                pq_flags: w.pq_flags,
                body: hex::decode(&w.body).map_err(D::Error::custom)?,
                sender_x: hex_arr(&w.sender_x).map_err(D::Error::custom)?,
                recipient_x: hex_arr(&w.recipient_x).map_err(D::Error::custom)?,
                outpoint: hex_arr(&w.outpoint).map_err(D::Error::custom)?,
            })
        }
    }
}

/// Recipient side: open a pq-layered directed-private body sent to me by
/// `locked.sender_x`. `mlkem_secret`/`password` are required exactly when
/// `locked.pq_flags` carries the corresponding bit — [`Error::NeedsMlKemKey`]
/// / [`Error::NeedsPassword`] otherwise. A supplied-but-wrong secret
/// (wrong keypair, wrong password) is indistinguishable from tampering —
/// both surface as [`Error::DecryptFailed`] (ML-KEM implicit rejection: a
/// wrong `MlKemSecret` still "succeeds" structurally, yielding a
/// pseudorandom-but-wrong shared secret that then fails the AEAD tag).
pub fn unlock_received(
    locked: &LockedBody,
    my_tweaked_seckey: &[u8; 32],
    mlkem_secret: Option<&MlKemSecret>,
    password: Option<&str>,
) -> Result<Vec<u8>, Error> {
    let mut body: &[u8] = &locked.body;
    let mut mlkem_ss: Option<[u8; 32]> = None;
    let mut alg_for_info: Option<MlKemAlg> = None;

    if locked.pq_flags & FLAG_MLKEM != 0 {
        let (prefix, rest) = parse_mlkem_prefix(body)?;
        body = rest;
        alg_for_info = Some(prefix.alg);
        let secret = mlkem_secret.ok_or(Error::NeedsMlKemKey)?;
        mlkem_ss = Some(decapsulate(prefix.alg, secret, prefix.ct)?);
    }

    let mut pw_key_bytes: Option<[u8; 32]> = None;
    if locked.pq_flags & FLAG_PW != 0 {
        let (params, rest) = parse_pw_prefix(body)?;
        body = rest;
        let pw = password.ok_or(Error::NeedsPassword)?;
        pw_key_bytes = Some(pw_key(pw, &params.salt, params.t, params.m_log2, params.p)?);
    }

    let shared_x = dm::ecdh_shared_x(my_tweaked_seckey, &locked.sender_x)?;
    let key =
        derive_pq_key(locked.pq_flags, alg_for_info, &shared_x, mlkem_ss.as_ref(), pw_key_bytes.as_ref());
    let aad = dm::dm_aad(&locked.sender_x, &locked.recipient_x, &locked.outpoint);
    crypt::open_aad(&key, &aad, body)
}

/// Sender re-reading their own sent note (wipe recovery — the `dm.rs`
/// `open_sent` analog). Possible ONLY when `locked.pq_flags == FLAG_PW`
/// alone: a KEM-layered note was encapsulated to the RECIPIENT's key, which
/// the sender never held, so [`Error::SenderCannotReopen`] is returned
/// whenever `FLAG_MLKEM` is set (with or without `FLAG_PW` alongside it) —
/// see the module doc.
pub fn unlock_sent(
    locked: &LockedBody,
    my_tweaked_seckey: &[u8; 32],
    my_output_x: &[u8; 32],
    password: Option<&str>,
) -> Result<Vec<u8>, Error> {
    if locked.pq_flags & FLAG_MLKEM != 0 {
        return Err(Error::SenderCannotReopen);
    }
    if locked.pq_flags & FLAG_PW == 0 {
        return Err(Error::Envelope("pq: locked body carries no reopenable layer"));
    }

    let (params, rest) = parse_pw_prefix(&locked.body)?;
    let pw = password.ok_or(Error::NeedsPassword)?;
    let pw_key_bytes = pw_key(pw, &params.salt, params.t, params.m_log2, params.p)?;

    let shared_x = dm::ecdh_shared_x(my_tweaked_seckey, &locked.recipient_x)?;
    let key = derive_pq_key(locked.pq_flags, None, &shared_x, None, Some(&pw_key_bytes));
    let aad = dm::dm_aad(my_output_x, &locked.recipient_x, &locked.outpoint);
    crypt::open_aad(&key, &aad, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A FLAG_PW note sealed under the ORIGINAL production params
    /// (t=3, m_log2=16, p=1 — what every PW note carried before the
    /// 2026-08-22 audit lowered [`PW_PROD_M_LOG2`] to 15) must still
    /// unlock: the decode cap ([`PW_MAX_M_LOG2`] = 16) deliberately
    /// admits the old value, and `pw_key`'s fallible-allocation refactor
    /// is byte-preserving (also pinned by `pw_key_vectors_are_pinned` in
    /// tests/pq.rs). Uses the private `derive_pq_key` to reconstruct the
    /// exact sealing the old emitter performed, since the public seal
    /// path now writes m_log2=15.
    #[test]
    fn old_prod_params_still_unlock() {
        let a = crate::bundle::Identity::from_app_seed(&[7u8; 32]).unwrap();
        let b = crate::bundle::Identity::from_app_seed(&[8u8; 32]).unwrap();
        let outpoint = [0x44u8; 36];
        let salt = [0x5au8; 16];
        let (t, m_log2, p) = (3u32, 16u8, 1u32);

        let pwk = pw_key("legacy m16 password", &salt, t, m_log2, p).unwrap();
        let shared = dm::ecdh_shared_x(&a.tweaked_seckey, &b.output_x).unwrap();
        let key = derive_pq_key(FLAG_PW, None, &shared, None, Some(&pwk));
        let aad = dm::dm_aad(&a.output_x, &b.output_x, &outpoint);
        let sealed = crypt::seal_aad(&key, &aad, b"pre-audit note").unwrap();

        let mut body = Vec::with_capacity(19 + sealed.len());
        body.extend_from_slice(&salt);
        body.push(t as u8);
        body.push(m_log2);
        body.push(p as u8);
        body.extend_from_slice(&sealed);
        let locked = LockedBody {
            pq_flags: FLAG_PW,
            body,
            sender_x: a.output_x,
            recipient_x: b.output_x,
            outpoint,
        };

        let pt = unlock_received(&locked, &b.tweaked_seckey, None, Some("legacy m16 password"))
            .unwrap();
        assert_eq!(pt, b"pre-audit note");
        // And the sender's own re-read path accepts the old params too.
        let pt2 =
            unlock_sent(&locked, &a.tweaked_seckey, &a.output_x, Some("legacy m16 password"))
                .unwrap();
        assert_eq!(pt2, b"pre-audit note");
    }
}
