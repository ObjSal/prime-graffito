//! Post-quantum sealing extension (notes-core/src/pq.rs) — ML-KEM/Argon2id
//! hybrid layers over the existing dm.rs ECDH for directed-private
//! single-recipient notes. See pq.rs's module doc for the wire format and
//! key-derivation domain.

use notes_core::address::Recipient;
use notes_core::bundle::{
    compose_directed_note_pq_exact_amount, compose_directed_note_pq_with_change_amount,
    extract_notes, extract_notes_pq, format_outpoint, Identity, OnchainTx, SyncBundle,
};
use notes_core::envelope::{self, FLAG_DIRECTED, FLAG_MLKEM, FLAG_MULTI, FLAG_PRIVATE, FLAG_PW};
use notes_core::pq::{
    self, export_private, export_public, fingerprint, import_private, import_public,
    mlkem_keypair_from_leaf, mlkem_seed_from_leaf, pw_key, seal_directed_pq, unlock_received,
    unlock_sent, LockedBody, MlKemAlg, MlKemKeypair, SealLayers,
};
use notes_core::tx::{op_return_payload, Utxo};
use notes_core::{Error, Network, DUST_LIMIT};

const NET: Network = Network::Regtest;
const AUX: [u8; 32] = [0x42; 32];
const ALL_ALGS: [MlKemAlg; 3] = [MlKemAlg::MlKem512, MlKemAlg::MlKem768, MlKemAlg::MlKem1024];

fn identity(byte: u8) -> Identity {
    Identity::from_app_seed(&[byte; 32]).unwrap()
}

fn utxos() -> Vec<Utxo> {
    vec![
        Utxo { txid: [1u8; 32], vout: 0, value: 60_000 },
        Utxo { txid: [2u8; 32], vout: 1, value: 25_000 },
        Utxo { txid: [3u8; 32], vout: 0, value: 1_000 },
    ]
}

/// Same shape as roundtrip.rs's private helper of the same name — a
/// companion-produced bundle from signed note txs.
fn bundle_from_txs(txs: &[(&notes_core::tx::NoteTx, bool, Option<u64>)]) -> SyncBundle {
    SyncBundle {
        network: "regtest".into(),
        notes_onchain: txs
            .iter()
            .map(|(note, self_spend, height)| OnchainTx {
                txid: note.txid_hex.clone(),
                height: *height,
                blocktime: height.map(|h| 1_700_000_000 + h),
                spends_from_self: *self_spend,
                payloads: note
                    .tx
                    .outputs
                    .iter()
                    .filter_map(|o| op_return_payload(&o.script_pubkey))
                    .map(hex::encode)
                    .collect(),
                pays_self: false,
                sender: None,
                author_candidates: Vec::new(),
                recipient: None,
                input_prevout_spks: Vec::new(),
                output_addrs: Vec::new(),
                first_input_outpoint: note
                    .spent_outpoints
                    .first()
                    .map(|(txid, vout)| format_outpoint(txid, *vout)),
            })
            .collect(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------
// 1. ML-KEM keypair basics: from_seed determinism + pinned vectors,
//    ek/ct/dk lengths, fingerprint format.
// ---------------------------------------------------------------------

fn fixed_seed() -> [u8; 64] {
    let mut s = [0u8; 64];
    for (i, b) in s.iter_mut().enumerate() {
        *b = i as u8;
    }
    s
}

#[test]
fn from_seed_is_deterministic_and_matches_pinned_vectors() {
    // Pinned first-32-hex-chars-of-ek vectors for the fixed seed
    // 0x00..0x3f, one per alg level (captured once from this exact
    // implementation — a regression catches any future drift in the
    // ml-kem dependency or this module's use of it).
    let vectors: [(MlKemAlg, &str); 3] = [
        (MlKemAlg::MlKem512, "3995815e597d104355cf29aa5333c932"),
        (MlKemAlg::MlKem768, "298aa10d423c8dda069d02bc59e6cdf0"),
        (MlKemAlg::MlKem1024, "4b94c29450111191823b3514c9ac1ea3"),
    ];
    let seed = fixed_seed();
    for (alg, want_prefix) in vectors {
        let a = MlKemKeypair::from_seed(alg, &seed);
        let b = MlKemKeypair::from_seed(alg, &seed);
        assert_eq!(a.ek(), b.ek(), "from_seed must be deterministic for {alg:?}");
        assert_eq!(a.ek().len(), alg.ek_len());
        assert_eq!(hex::encode(a.ek()).len(), alg.ek_len() * 2);
        let got_prefix = &hex::encode(a.ek())[..32];
        assert_eq!(got_prefix, want_prefix, "ek vector drifted for {alg:?}");
    }
}

#[test]
fn generate_draws_fresh_entropy() {
    // Two independent generate() calls must not collide (trivially true
    // for a 64-byte TRNG seed, but worth asserting as a sanity check that
    // generate() isn't secretly deterministic).
    let a = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let b = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    assert_ne!(a.ek(), b.ek());
    assert_ne!(a.seed(), b.seed());
}

#[test]
fn fingerprint_format_vector() {
    let seed = fixed_seed();
    let kp = MlKemKeypair::from_seed(MlKemAlg::MlKem768, &seed);
    let fp = kp.fingerprint();
    // "xxxx xxxx xxxx xxxx" — 16 lowercase hex chars in 4 groups of 4.
    let parts: Vec<&str> = fp.split(' ').collect();
    assert_eq!(parts.len(), 4, "fingerprint {fp:?} must have 4 space-separated groups");
    for p in &parts {
        assert_eq!(p.len(), 4);
        assert!(p.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
    assert_eq!(fp.replace(' ', "").len(), 16);
    // Deterministic + matches the free function directly.
    assert_eq!(fp, fingerprint(MlKemAlg::MlKem768, kp.ek()));
    // Different alg (even with the same ek bytes conceptually can't
    // happen, but different seed certainly) -> different fingerprint.
    let other = MlKemKeypair::from_seed(MlKemAlg::MlKem768, &[9u8; 64]);
    assert_ne!(fp, other.fingerprint());
}

#[test]
fn alg_id_and_lengths_round_trip() {
    for alg in ALL_ALGS {
        assert_eq!(MlKemAlg::from_id(alg.id()), Some(alg));
    }
    assert_eq!(MlKemAlg::from_id(0x00), None);
    assert_eq!(MlKemAlg::from_id(0x04), None);
    assert_eq!(MlKemAlg::from_id(0xff), None);
}

// ---------------------------------------------------------------------
// 2. Decapsulation-secret parity: Seed vs Expanded of the same keypair.
// ---------------------------------------------------------------------

#[test]
fn decap_parity_seed_vs_expanded() {
    let sender = identity(1);
    let recipient = identity(2);
    let outpoint = [0x21u8; 36];
    for alg in ALL_ALGS {
        let kp = MlKemKeypair::generate(alg).unwrap();
        let layers = SealLayers { mlkem_ek: Some((alg, kp.ek())), password: None };
        let (pq_flags, body) = seal_directed_pq(
            &sender.tweaked_seckey,
            &sender.output_x,
            &recipient.output_x,
            &outpoint,
            b"parity check",
            layers,
        )
        .unwrap();
        let locked = LockedBody {
            pq_flags,
            body,
            sender_x: sender.output_x,
            recipient_x: recipient.output_x,
            outpoint,
        };
        let via_seed =
            unlock_received(&locked, &recipient.tweaked_seckey, Some(&kp.secret()), None).unwrap();
        let via_expanded =
            unlock_received(&locked, &recipient.tweaked_seckey, Some(&kp.expanded_secret()), None)
                .unwrap();
        assert_eq!(via_seed, b"parity check");
        assert_eq!(via_seed, via_expanded, "Seed and Expanded must recover identical plaintext for {alg:?}");
    }
}

// ---------------------------------------------------------------------
// 3. Armor round trip + rejection cases.
// ---------------------------------------------------------------------

#[test]
fn armor_round_trip_all_levels_private_and_public() {
    for alg in ALL_ALGS {
        let kp = MlKemKeypair::generate(alg).unwrap();

        let priv_armor = export_private(alg, kp.seed());
        assert!(priv_armor.contains("BEGIN GRAFFITO ML-KEM PRIVATE KEY"));
        assert!(priv_armor.contains("END GRAFFITO ML-KEM PRIVATE KEY"));
        let (alg2, seed2) = import_private(&priv_armor).unwrap();
        assert_eq!(alg2, alg);
        assert_eq!(&seed2, kp.seed());

        let pub_armor = export_public(alg, kp.ek());
        assert!(pub_armor.contains("BEGIN GRAFFITO ML-KEM PUBLIC KEY"));
        assert!(pub_armor.contains("END GRAFFITO ML-KEM PUBLIC KEY"));
        let (alg3, ek3) = import_public(&pub_armor).unwrap();
        assert_eq!(alg3, alg);
        assert_eq!(ek3, kp.ek());

        // Liberal about whitespace/line length: collapse all newlines and
        // re-indent — still parses.
        let squished = priv_armor.replace('\n', "   \n  ");
        let (alg4, seed4) = import_private(&squished).unwrap();
        assert_eq!(alg4, alg);
        assert_eq!(&seed4, kp.seed());
    }
}

#[test]
fn armor_rejects_bad_shapes() {
    let kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let priv_armor = export_private(MlKemAlg::MlKem768, kp.seed());
    let pub_armor = export_public(MlKemAlg::MlKem768, kp.ek());

    // Wrong label entirely.
    assert!(import_private("-----BEGIN NOT A KEY-----\nAAAA\n-----END NOT A KEY-----\n").is_err());
    // Public armor fed to the private importer (wrong label) and vice versa.
    assert!(import_private(&pub_armor).is_err());
    assert!(import_public(&priv_armor).is_err());

    // Wrong version byte: re-armor a payload with version 0x02.
    let mut payload = vec![0x02u8, MlKemAlg::MlKem768.id()];
    payload.extend_from_slice(kp.seed());
    let bad_version = format!(
        "-----BEGIN GRAFFITO ML-KEM PRIVATE KEY-----\n{}\n-----END GRAFFITO ML-KEM PRIVATE KEY-----\n",
        {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(&payload)
        }
    );
    assert!(import_private(&bad_version).is_err());

    // Truncated payload (private key body cut short).
    let truncated: String = priv_armor.chars().take(priv_armor.len() - 40).collect();
    assert!(import_private(&truncated).is_err());

    // Wrong ek length for the claimed alg (truncate a public armor's ek).
    let short_pub = export_public(MlKemAlg::MlKem768, &kp.ek()[..kp.ek().len() - 10]);
    assert!(import_public(&short_pub).is_err());
}

// ---------------------------------------------------------------------
// 4. Seal -> unlock round trip for every layer combination, recipient side.
// ---------------------------------------------------------------------

#[test]
fn seal_unlock_round_trip_all_layer_combinations() {
    let sender = identity(3);
    let recipient = identity(4);
    let outpoint = [0x55u8; 36];
    let plaintext = b"round trip across every combo";

    struct Combo {
        name: &'static str,
        alg: Option<MlKemAlg>,
        password: Option<&'static str>,
    }
    let combos = [
        Combo { name: "pw only", alg: None, password: Some("correct horse battery staple") },
        Combo { name: "kem512 only", alg: Some(MlKemAlg::MlKem512), password: None },
        Combo { name: "kem768 only", alg: Some(MlKemAlg::MlKem768), password: None },
        Combo { name: "kem1024 only", alg: Some(MlKemAlg::MlKem1024), password: None },
        Combo { name: "kem768 + pw", alg: Some(MlKemAlg::MlKem768), password: Some("hybrid layer") },
    ];

    for combo in combos {
        let kp = combo.alg.map(MlKemKeypair::generate).transpose().unwrap();
        let layers = SealLayers {
            mlkem_ek: kp.as_ref().map(|k| (k.alg(), k.ek())),
            password: combo.password,
        };
        let expect_flags = (if combo.alg.is_some() { FLAG_MLKEM } else { 0 })
            | (if combo.password.is_some() { FLAG_PW } else { 0 });

        let (pq_flags, body) = seal_directed_pq(
            &sender.tweaked_seckey,
            &sender.output_x,
            &recipient.output_x,
            &outpoint,
            plaintext,
            layers,
        )
        .unwrap();
        assert_eq!(pq_flags, expect_flags, "{}", combo.name);

        let locked = LockedBody {
            pq_flags,
            body,
            sender_x: sender.output_x,
            recipient_x: recipient.output_x,
            outpoint,
        };
        let secret = kp.as_ref().map(MlKemKeypair::secret);
        let pt = unlock_received(&locked, &recipient.tweaked_seckey, secret.as_ref(), combo.password)
            .unwrap();
        assert_eq!(pt, plaintext, "{}", combo.name);
    }
}

#[test]
fn seal_directed_pq_requires_at_least_one_layer() {
    let sender = identity(5);
    let recipient = identity(6);
    let layers = SealLayers { mlkem_ek: None, password: None };
    let err = seal_directed_pq(
        &sender.tweaked_seckey,
        &sender.output_x,
        &recipient.output_x,
        &[0u8; 36],
        b"x",
        layers,
    )
    .unwrap_err();
    assert!(matches!(err, Error::Envelope(_)));
}

// ---------------------------------------------------------------------
// 5. unlock_sent: pw-only works, kem (with or without pw) is
//    SenderCannotReopen.
// ---------------------------------------------------------------------

#[test]
fn unlock_sent_pw_only_succeeds_kem_always_fails() {
    let sender = identity(7);
    let recipient = identity(8);
    let outpoint = [0x66u8; 36];

    // pw-only: sender CAN re-open their own sent note.
    let layers = SealLayers { mlkem_ek: None, password: Some("wipe recovery phrase") };
    let (pq_flags, body) = seal_directed_pq(
        &sender.tweaked_seckey, &sender.output_x, &recipient.output_x, &outpoint,
        b"sender re-read", layers,
    )
    .unwrap();
    let locked_pw = LockedBody {
        pq_flags, body, sender_x: sender.output_x, recipient_x: recipient.output_x, outpoint,
    };
    let pt = unlock_sent(
        &locked_pw, &sender.tweaked_seckey, &sender.output_x, Some("wipe recovery phrase"),
    )
    .unwrap();
    assert_eq!(pt, b"sender re-read");

    // kem-only and kem+pw: sender can NEVER re-open (SenderCannotReopen),
    // regardless of whether they also supply the correct password.
    for password in [None, Some("hybrid layer")] {
        let kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
        let layers = SealLayers { mlkem_ek: Some((kp.alg(), kp.ek())), password };
        let (pq_flags, body) = seal_directed_pq(
            &sender.tweaked_seckey, &sender.output_x, &recipient.output_x, &outpoint,
            b"never reopenable by sender", layers,
        )
        .unwrap();
        let locked = LockedBody {
            pq_flags, body, sender_x: sender.output_x, recipient_x: recipient.output_x, outpoint,
        };
        let err = unlock_sent(&locked, &sender.tweaked_seckey, &sender.output_x, password)
            .unwrap_err();
        assert!(matches!(err, Error::SenderCannotReopen), "password={password:?}");
        // But the recipient reads it fine.
        let pt = unlock_received(&locked, &recipient.tweaked_seckey, Some(&kp.secret()), password)
            .unwrap();
        assert_eq!(pt, b"never reopenable by sender");
    }
}

// ---------------------------------------------------------------------
// 6. Error precision: wrong password/key, missing password/key.
// ---------------------------------------------------------------------

#[test]
fn wrong_and_missing_secrets_are_reported_precisely() {
    let sender = identity(9);
    let recipient = identity(10);
    let outpoint = [0x77u8; 36];

    // --- password layer ---
    let layers = SealLayers { mlkem_ek: None, password: Some("the real password") };
    let (pq_flags, body) = seal_directed_pq(
        &sender.tweaked_seckey, &sender.output_x, &recipient.output_x, &outpoint,
        b"pw secrets", layers,
    )
    .unwrap();
    let locked = LockedBody {
        pq_flags, body, sender_x: sender.output_x, recipient_x: recipient.output_x, outpoint,
    };
    let wrong = unlock_received(&locked, &recipient.tweaked_seckey, None, Some("nope")).unwrap_err();
    assert!(matches!(wrong, Error::DecryptFailed));
    let missing = unlock_received(&locked, &recipient.tweaked_seckey, None, None).unwrap_err();
    assert!(matches!(missing, Error::NeedsPassword));

    // --- kem layer ---
    let kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let other_kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let layers = SealLayers { mlkem_ek: Some((kp.alg(), kp.ek())), password: None };
    let (pq_flags, body) = seal_directed_pq(
        &sender.tweaked_seckey, &sender.output_x, &recipient.output_x, &outpoint,
        b"kem secrets", layers,
    )
    .unwrap();
    let locked_kem = LockedBody {
        pq_flags, body, sender_x: sender.output_x, recipient_x: recipient.output_x, outpoint,
    };
    let missing_kem =
        unlock_received(&locked_kem, &recipient.tweaked_seckey, None, None).unwrap_err();
    assert!(matches!(missing_kem, Error::NeedsMlKemKey));
    // A DIFFERENT keypair's secret is structurally compatible (same alg,
    // same lengths) but cryptographically wrong -> ML-KEM implicit
    // rejection means this surfaces as DecryptFailed, not a distinct
    // "wrong key" error.
    let wrong_kem = unlock_received(
        &locked_kem, &recipient.tweaked_seckey, Some(&other_kp.secret()), None,
    )
    .unwrap_err();
    assert!(matches!(wrong_kem, Error::DecryptFailed));

    // A structurally-incompatible (wrong alg's) Expanded secret is caught
    // before any crypto runs.
    let kp1024 = MlKemKeypair::generate(MlKemAlg::MlKem1024).unwrap();
    let mismatched = unlock_received(
        &locked_kem, &recipient.tweaked_seckey, Some(&kp1024.expanded_secret()), None,
    )
    .unwrap_err();
    assert!(matches!(mismatched, Error::MlKemAlgMismatch));
}

// ---------------------------------------------------------------------
// 7. Tamper tests: (a) kem ct, (b) pw params block, (c) sealed blob.
// ---------------------------------------------------------------------

#[test]
fn tampering_any_section_breaks_the_open() {
    let sender = identity(11);
    let recipient = identity(12);
    let outpoint = [0x88u8; 36];
    let kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let layers =
        SealLayers { mlkem_ek: Some((kp.alg(), kp.ek())), password: Some("tamper test pw") };
    let (pq_flags, body) = seal_directed_pq(
        &sender.tweaked_seckey, &sender.output_x, &recipient.output_x, &outpoint,
        b"tamper me not", layers,
    )
    .unwrap();
    assert_eq!(pq_flags, FLAG_MLKEM | FLAG_PW);

    // Layout: alg_id(1) || ct(ct_len) || salt(16) || t(1) || m_log2(1) ||
    // p(1) || sealed_blob.
    let ct_len = MlKemAlg::MlKem768.ct_len();
    let pw_offset = 1 + ct_len;
    let blob_offset = pw_offset + 19;
    assert!(body.len() > blob_offset, "sanity: body must contain a sealed blob");

    let open = |b: &[u8]| {
        let locked = LockedBody {
            pq_flags,
            body: b.to_vec(),
            sender_x: sender.output_x,
            recipient_x: recipient.output_x,
            outpoint,
        };
        unlock_received(
            &locked, &recipient.tweaked_seckey, Some(&kp.secret()), Some("tamper test pw"),
        )
    };

    // Sanity: untampered body opens.
    assert_eq!(open(&body).unwrap(), b"tamper me not");

    // (a) flip a byte inside the KEM ciphertext.
    let mut tampered_ct = body.clone();
    tampered_ct[1 + 5] ^= 0xff;
    assert!(matches!(open(&tampered_ct).unwrap_err(), Error::DecryptFailed), "kem ct tamper");

    // (b) flip a byte inside the pw params block (the salt, so t/m_log2/p
    // stay structurally valid and the failure is cryptographic, not a
    // parameter-validation error).
    let mut tampered_pw = body.clone();
    tampered_pw[pw_offset + 2] ^= 0xff;
    assert!(matches!(open(&tampered_pw).unwrap_err(), Error::DecryptFailed), "pw params tamper");

    // (c) flip a byte inside the sealed blob (past the nonce, in the
    // ciphertext/tag region).
    let mut tampered_blob = body.clone();
    let last = tampered_blob.len() - 1;
    tampered_blob[last] ^= 0xff;
    assert!(matches!(open(&tampered_blob).unwrap_err(), Error::DecryptFailed), "sealed blob tamper");
}

// ---------------------------------------------------------------------
// 8. Envelope: pq flag validity and the v1 regression vector.
// ---------------------------------------------------------------------

#[test]
fn envelope_pq_flags_decode_when_valid() {
    // FLAG_PRIVATE|FLAG_DIRECTED|FLAG_PW (0x01|0x02|0x10 = 0x13).
    let flags = FLAG_PRIVATE | FLAG_DIRECTED | FLAG_PW;
    let payloads = envelope::encode_outputs(flags, None, b"pw-sealed body", 200).unwrap();
    let decoded = envelope::decode_note(&payloads).unwrap();
    assert_eq!(decoded.flags, flags);
    assert_eq!(decoded.body, b"pw-sealed body");

    // FLAG_PRIVATE|FLAG_DIRECTED|FLAG_MLKEM (0x01|0x02|0x20 = 0x23).
    let flags = FLAG_PRIVATE | FLAG_DIRECTED | FLAG_MLKEM;
    let payloads = envelope::encode_outputs(flags, None, b"kem-sealed body", 200).unwrap();
    let decoded = envelope::decode_note(&payloads).unwrap();
    assert_eq!(decoded.flags, flags);

    // Both together.
    let flags = FLAG_PRIVATE | FLAG_DIRECTED | FLAG_PW | FLAG_MLKEM;
    let payloads = envelope::encode_outputs(flags, None, b"hybrid body", 200).unwrap();
    let decoded = envelope::decode_note(&payloads).unwrap();
    assert_eq!(decoded.flags, flags);
}

#[test]
fn envelope_pq_flag_without_private_directed_is_undecodable() {
    // FLAG_PW alone (no FLAG_PRIVATE, no FLAG_DIRECTED): header-level
    // encode_outputs must reject it...
    assert!(envelope::encode_outputs(FLAG_PW, None, b"x", 80).is_err());
    // ...and a hand-built header (bypassing the encoder) must decode to
    // None, matching the FLAG_MULTI-without-FLAG_DIRECTED guard.
    assert!(envelope::decode_note(&[b"PNTE110 hi".to_vec()]).is_none());
    // FLAG_PRIVATE set but not FLAG_DIRECTED is still not enough.
    assert!(envelope::decode_note(&[b"PNTE111 hi".to_vec()]).is_none());
    // FLAG_MLKEM alone, same story.
    assert!(envelope::encode_outputs(FLAG_MLKEM, None, b"x", 80).is_err());
    assert!(envelope::decode_note(&[b"PNTE120 hi".to_vec()]).is_none());
}

#[test]
fn envelope_pq_flags_incompatible_with_multi() {
    let flags = FLAG_PRIVATE | FLAG_DIRECTED | FLAG_MULTI | FLAG_PW;
    assert_eq!(flags, 0x17);
    assert!(envelope::encode_outputs(flags, Some(2), b"xx", 80).is_err());
    // Header hand-built: "PNTE" + '1' (version) + flags hex("17") + count
    // hex("02") + ' '.
    assert!(envelope::decode_note(&[b"PNTE11702 hi".to_vec()]).is_none());

    let flags2 = FLAG_PRIVATE | FLAG_DIRECTED | FLAG_MULTI | FLAG_MLKEM;
    assert_eq!(flags2, 0x27);
    assert!(envelope::encode_outputs(flags2, Some(2), b"xx", 80).is_err());
    assert!(envelope::decode_note(&[b"PNTE12702 hi".to_vec()]).is_none());
}

#[test]
fn envelope_v1_note_still_round_trips_byte_identically() {
    // Regression: an ordinary v1 directed-private note (flags = 0x03,
    // FLAG_PRIVATE|FLAG_DIRECTED, no pq bits) must decode exactly as
    // before — pq's validate_pq is a no-op when neither pq bit is set.
    let flags = FLAG_PRIVATE | FLAG_DIRECTED;
    assert_eq!(flags, 0x03);
    let body = b"a perfectly ordinary v1 body".to_vec();
    let payloads = envelope::encode_outputs(flags, None, &body, 80).unwrap();
    assert_eq!(payloads[0][..8], *b"PNTE103 ");
    let decoded = envelope::decode_note(&payloads).unwrap();
    assert_eq!(decoded.flags, 0x03);
    assert_eq!(decoded.multi_count, None);
    assert_eq!(decoded.body, body);
}

// ---------------------------------------------------------------------
// 9. End-to-end through bundle::compose_directed_note_pq_* + extraction.
// ---------------------------------------------------------------------

#[test]
fn compose_pq_note_extracts_locked_then_unlocks() {
    let a = identity(20); // sender
    let b = identity(21); // recipient
    let to_b = Recipient::parse(NET, &b.address(NET)).unwrap();
    let kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let layers = SealLayers { mlkem_ek: Some((kp.alg(), kp.ek())), password: Some("compose e2e") };

    let note = compose_directed_note_pq_with_change_amount(
        &a, &utxos(), "pq note for bob", &to_b, DUST_LIMIT, layers, None, 80, 1.0, 0, || Ok(AUX),
    )
    .unwrap();

    // Recipient side: bundle as the companion would build it (B did not
    // spend, the tx pays B's address, author = A).
    let mut recv_bundle = bundle_from_txs(&[(&note, false, Some(100))]);
    recv_bundle.notes_onchain[0].pays_self = true;
    recv_bundle.notes_onchain[0].sender = Some(a.address(NET));
    let recv_notes = extract_notes(&recv_bundle, &b, NET);
    assert_eq!(recv_notes.len(), 1);
    let recv_note = &recv_notes[0];
    assert!(recv_note.private && recv_note.directed && recv_note.received);
    assert_eq!(recv_note.pq_flags, FLAG_MLKEM | FLAG_PW);
    assert!(recv_note.text.is_none(), "pq notes never auto-decrypt via extract_notes");
    let locked = recv_note.locked.clone().expect("keyed scan must populate LockedBody");
    assert_eq!(locked.pq_flags, FLAG_MLKEM | FLAG_PW);
    assert_eq!(locked.sender_x, a.output_x);
    assert_eq!(locked.recipient_x, b.output_x);

    let pt = unlock_received(
        &locked, &b.tweaked_seckey, Some(&kp.secret()), Some("compose e2e"),
    )
    .unwrap();
    assert_eq!(pt, b"pq note for bob");

    // Sender side: A re-scans their own sent tx.
    let mut sent_bundle = bundle_from_txs(&[(&note, true, Some(100))]);
    sent_bundle.notes_onchain[0].recipient = Some(b.address(NET));
    let sent_notes = extract_notes(&sent_bundle, &a, NET);
    assert_eq!(sent_notes.len(), 1);
    let sent_note = &sent_notes[0];
    assert!(!sent_note.received && sent_note.directed && sent_note.private);
    assert_eq!(sent_note.pq_flags, FLAG_MLKEM | FLAG_PW);
    let sent_locked = sent_note.locked.clone().expect("keyed scan must populate LockedBody");
    assert_eq!(sent_locked.sender_x, a.output_x);
    assert_eq!(sent_locked.recipient_x, b.output_x);
    // Because FLAG_MLKEM is set, the sender can never reopen it themselves.
    let err = unlock_sent(&sent_locked, &a.tweaked_seckey, &a.output_x, Some("compose e2e"))
        .unwrap_err();
    assert!(matches!(err, Error::SenderCannotReopen));
}

#[test]
fn extract_notes_pq_auto_unlocks_kem_only_received_note() {
    let a = identity(22);
    let b = identity(23);
    let to_b = Recipient::parse(NET, &b.address(NET)).unwrap();
    let kp = MlKemKeypair::generate(MlKemAlg::MlKem512).unwrap();
    let layers = SealLayers { mlkem_ek: Some((kp.alg(), kp.ek())), password: None };

    let note = compose_directed_note_pq_with_change_amount(
        &a, &utxos(), "auto-unlock me", &to_b, DUST_LIMIT, layers, None, 80, 1.0, 0, || Ok(AUX),
    )
    .unwrap();
    let mut bundle = bundle_from_txs(&[(&note, false, Some(200))]);
    bundle.notes_onchain[0].pays_self = true;
    bundle.notes_onchain[0].sender = Some(a.address(NET));

    // Plain extract_notes leaves it locked...
    let plain = extract_notes(&bundle, &b, NET);
    assert_eq!(plain[0].text, None);
    assert_eq!(plain[0].pq_flags, FLAG_MLKEM);

    // ...extract_notes_pq, given B's secret, auto-unlocks it.
    let secrets = vec![kp.secret()];
    let self_spk = notes_core::address::p2tr_script_pubkey(&b.output_x);
    let unlocked = extract_notes_pq(&bundle, &b, NET, &[self_spk], &[], &secrets);
    assert_eq!(unlocked.len(), 1);
    assert_eq!(unlocked[0].text.as_deref(), Some("auto-unlock me"));
    assert_eq!(unlocked[0].pq_flags, FLAG_MLKEM);

    // A secret that doesn't match anything present leaves it locked, not
    // panicking or silently "succeeding".
    let wrong_kp = MlKemKeypair::generate(MlKemAlg::MlKem512).unwrap();
    let wrong_secrets = vec![wrong_kp.secret()];
    let still_locked = extract_notes_pq(&bundle, &b, NET, &[], &[], &wrong_secrets);
    assert_eq!(still_locked[0].text, None);
}

#[test]
fn extract_notes_pq_never_attempts_own_kem_note() {
    // An own (sent) kem-layered note must stay locked forever from the
    // sender's own extract_notes_pq scan — attempting it would be both
    // wasted work and (per unlock_sent) structurally impossible.
    let a = identity(24);
    let b = identity(25);
    let to_b = Recipient::parse(NET, &b.address(NET)).unwrap();
    let kp = MlKemKeypair::generate(MlKemAlg::MlKem768).unwrap();
    let layers = SealLayers { mlkem_ek: Some((kp.alg(), kp.ek())), password: None };
    let note = compose_directed_note_pq_with_change_amount(
        &a, &utxos(), "own kem note", &to_b, DUST_LIMIT, layers, None, 80, 1.0, 0, || Ok(AUX),
    )
    .unwrap();
    let mut bundle = bundle_from_txs(&[(&note, true, Some(300))]);
    bundle.notes_onchain[0].recipient = Some(b.address(NET));

    let secrets = vec![kp.secret()]; // even if A somehow held it
    let notes = extract_notes_pq(&bundle, &a, NET, &[], &[], &secrets);
    assert_eq!(notes.len(), 1);
    assert!(!notes[0].received);
    assert_eq!(notes[0].text, None, "own kem note must never auto-unlock");
}

#[test]
fn compose_directed_note_pq_exact_amount_coin_control() {
    let a = identity(26);
    let b = identity(27);
    let to_b = Recipient::parse(NET, &b.address(NET)).unwrap();
    let inputs = utxos();
    let layers = SealLayers { mlkem_ek: None, password: Some("exact amount pw") };
    let note = compose_directed_note_pq_exact_amount(
        &a, &inputs, "exact inputs pq", &to_b, DUST_LIMIT, layers, None, 80, 1.0, 0, || Ok(AUX),
    )
    .unwrap();
    // Spends exactly every given coin (coin control contract).
    assert_eq!(note.tx.inputs.len(), inputs.len());

    let mut bundle = bundle_from_txs(&[(&note, false, Some(400))]);
    bundle.notes_onchain[0].pays_self = true;
    bundle.notes_onchain[0].sender = Some(a.address(NET));
    let notes = extract_notes(&bundle, &b, NET);
    assert_eq!(notes[0].pq_flags, FLAG_PW);
    let locked = notes[0].locked.clone().unwrap();
    let pt =
        unlock_received(&locked, &b.tweaked_seckey, None, Some("exact amount pw")).unwrap();
    assert_eq!(pt, b"exact inputs pq");
}

// ---------------------------------------------------------------------
// 10. Argon2 production parameters exercised at least once (explicit,
//     beyond the many implicit uses above via seal_directed_pq).
// ---------------------------------------------------------------------

#[test]
fn argon2_production_params_direct() {
    let salt = [0x5au8; 16];
    let out = pw_key("production parameter check", &salt, pq::PW_PROD_T, pq::PW_PROD_M_LOG2, pq::PW_PROD_P).unwrap();
    assert_eq!(out.len(), 32);
    // Deterministic for the same inputs.
    let out2 = pw_key("production parameter check", &salt, pq::PW_PROD_T, pq::PW_PROD_M_LOG2, pq::PW_PROD_P).unwrap();
    assert_eq!(out, out2);
    // Different password -> different key.
    let out3 = pw_key("different password", &salt, pq::PW_PROD_T, pq::PW_PROD_M_LOG2, pq::PW_PROD_P).unwrap();
    assert_ne!(out, out3);
    // Production params must always sit within the decode caps — a prod
    // bump past the caps would emit notes every peer rejects (the
    // raise-the-cap-first rule on PW_MAX_M_LOG2/PW_MAX_T).
    assert!(pq::PW_PROD_M_LOG2 <= pq::PW_MAX_M_LOG2);
    assert!(pq::PW_PROD_T <= pq::PW_MAX_T);
}

/// Guards the 2026-08-22 fallible-allocation refactor (audit F2) and any
/// future argon2 crate bump: `pw_key` must stay byte-identical at BOTH the
/// original (m_log2=16) and current (15) production params — vectors
/// captured from the pre-refactor implementation. A mismatch means
/// already-sealed FLAG_PW notes stop unlocking: SHIP-BLOCKING, never
/// "fix the hex".
#[test]
fn pw_key_vectors_are_pinned() {
    let salt = [0x5au8; 16];
    let v16 = pw_key("graffito pinned vector", &salt, 3, 16, 1).unwrap();
    assert_eq!(
        hex::encode(v16),
        "fad9b05640e82dc3b10856d820ee43b57029b9ec5233d378a16005fa57856e73"
    );
    let v15 = pw_key("graffito pinned vector", &salt, 3, 15, 1).unwrap();
    assert_eq!(
        hex::encode(v15),
        "cf62ed0e748cd55190ce559438fc3af03d477ab106bf79176c66fa1125ae589e"
    );
}

/// 2026-08-22 audit F1: `(t, m_log2, p)` are attacker-controlled on-chain
/// bytes and Argon2 allocates `2^m_log2` KiB up front, so the decode caps
/// must reject a hostile demand BEFORE any allocation (`Error::Decode`).
/// Reaching `DecryptFailed` instead proves the params were ACCEPTED — the
/// arena was allocated and the AEAD ran — which is asserted for the
/// boundary values so the cap sits exactly where it claims.
#[test]
fn hostile_argon2_params_are_rejected_at_decode() {
    let a = identity(31);
    let b = identity(32);
    let outpoint = [0x77u8; 36];
    let locked = |t: u8, m_log2: u8, p: u8| {
        let mut body = vec![0u8; 16]; // salt
        body.push(t);
        body.push(m_log2);
        body.push(p);
        body.extend_from_slice(&[0u8; 64]); // fake sealed blob
        LockedBody {
            pq_flags: FLAG_PW,
            body,
            sender_x: a.output_x,
            recipient_x: b.output_x,
            outpoint,
        }
    };
    let unlock =
        |t, m, p| unlock_received(&locked(t, m, p), &b.tweaked_seckey, None, Some("pw"));

    // The pre-audit cap admitted up to 24 (a 16 GiB arena). Now: > 16 is
    // undecodable, as are Argon2's own out-of-range minimums.
    assert!(matches!(unlock(3, 24, 1).unwrap_err(), Error::Decode(_)), "m_log2=24 (16 GiB)");
    assert!(matches!(unlock(3, 17, 1).unwrap_err(), Error::Decode(_)), "m_log2=17");
    assert!(matches!(unlock(17, 15, 1).unwrap_err(), Error::Decode(_)), "t=17");
    assert!(matches!(unlock(0, 15, 1).unwrap_err(), Error::Decode(_)), "t=0");
    assert!(matches!(unlock(3, 15, 0).unwrap_err(), Error::Decode(_)), "p=0");

    // Boundary acceptance: m_log2=16 (the original prod value — every
    // pre-audit FLAG_PW note) and t=16 pass validation and fail only at
    // the AEAD tag (the blob here is garbage).
    assert!(
        matches!(unlock(3, 16, 1).unwrap_err(), Error::DecryptFailed),
        "m_log2=16 must stay decodable"
    );
    assert!(matches!(unlock(16, 15, 1).unwrap_err(), Error::DecryptFailed), "t=16");
}

#[test]
fn pq_overhead_matches_actual_body_growth() {
    let sender = identity(28);
    let recipient = identity(29);
    let outpoint = [0x99u8; 36];
    for alg in ALL_ALGS {
        let kp = MlKemKeypair::generate(alg).unwrap();
        let layers = SealLayers { mlkem_ek: Some((alg, kp.ek())), password: None };
        let plaintext = b"overhead check";
        let (pq_flags, body) = seal_directed_pq(
            &sender.tweaked_seckey, &sender.output_x, &recipient.output_x, &outpoint, plaintext,
            layers,
        )
        .unwrap();
        let expected_len =
            plaintext.len() + notes_core::crypt::SEAL_OVERHEAD + pq::pq_overhead(pq_flags, Some(alg));
        assert_eq!(body.len(), expected_len, "{alg:?}");
    }

    // pw-only overhead is exactly 19 bytes, independent of alg.
    assert_eq!(pq::pq_overhead(FLAG_PW, None), 19);
    assert_eq!(pq::pq_overhead(0, None), 0);
}

// ---------------------------------------------------------------------
// Seed-derived receive keypair (pq.rs's "Seed-derived receive keypair"
// section) — the cross-app recovery contract shared with the Mac app's
// `app-core/src/pqkeys.rs`. Determinism/independence checks mirror the
// Mac app's `pqkeys` unit tests; the pinned vectors below are copied
// LITERALLY from that suite's `pinned_derivation_vectors_per_level` so a
// mismatch here means the two apps would derive different keys from the
// same notebook — SHIP-BLOCKING, never "fix" it by updating the hex.
// ---------------------------------------------------------------------

fn leaf(byte: u8) -> [u8; 32] {
    [byte; 32]
}

#[test]
fn mlkem_keypair_from_leaf_is_deterministic() {
    let l = leaf(0x11);
    let a = mlkem_keypair_from_leaf(&l, MlKemAlg::MlKem768);
    let b = mlkem_keypair_from_leaf(&l, MlKemAlg::MlKem768);
    assert_eq!(a.ek(), b.ek());
    assert_eq!(a.seed(), b.seed());
}

#[test]
fn mlkem_keypair_from_leaf_different_leaf_secrets_differ() {
    let a = mlkem_keypair_from_leaf(&leaf(0x11), MlKemAlg::MlKem768);
    let b = mlkem_keypair_from_leaf(&leaf(0x22), MlKemAlg::MlKem768);
    assert_ne!(a.ek(), b.ek());
    assert_ne!(a.seed(), b.seed());
}

#[test]
fn mlkem_keypair_from_leaf_different_levels_are_independent_draws() {
    // Same leaf_secret, three levels: the seeds must differ too, proving
    // the alg id genuinely folds into the HKDF info rather than the
    // three levels sharing one expansion.
    let l = leaf(0x33);
    let k512 = mlkem_keypair_from_leaf(&l, MlKemAlg::MlKem512);
    let k768 = mlkem_keypair_from_leaf(&l, MlKemAlg::MlKem768);
    let k1024 = mlkem_keypair_from_leaf(&l, MlKemAlg::MlKem1024);
    assert_ne!(k512.seed(), k768.seed());
    assert_ne!(k768.seed(), k1024.seed());
    assert_ne!(k512.seed(), k1024.seed());
}

#[test]
fn mlkem_seed_from_leaf_matches_keypair_from_leaf_seed() {
    let l = leaf(0x44);
    for alg in ALL_ALGS {
        let seed = mlkem_seed_from_leaf(&l, alg);
        let kp = mlkem_keypair_from_leaf(&l, alg);
        assert_eq!(&*seed, kp.seed());
    }
}

// ---- FROZEN cross-app derivation vectors ----------------------------
//
// Same fixed leaf_secret and expected ek-prefix hex as the Mac app's
// `app-core/src/pqkeys.rs` `pinned_derivation_vectors_per_level` test —
// literal-for-literal. If this ever fails, the derivation changed and
// every already-shared pq receive key would silently stop matching what
// a contact has stored. Do not "fix" this test by updating the hex;
// treat a failure here as SHIP-BLOCKING.

const FIXED_LEAF: [u8; 32] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
    0x1f, 0x20,
];

#[test]
fn pinned_derivation_vectors_per_level() {
    for (alg, expected_ek_prefix_hex, expected_len) in [
        (MlKemAlg::MlKem512, "0b629735573ce73b363d2acc1ad998c4", 800usize),
        (MlKemAlg::MlKem768, "7cbbcf17294071d7674ffb5618848fa5", 1184usize),
        (MlKemAlg::MlKem1024, "7f7a36d7b029f0088dc9e970d5304f5f", 1568usize),
    ] {
        let kp = mlkem_keypair_from_leaf(&FIXED_LEAF, alg);
        assert_eq!(kp.ek().len(), expected_len);
        let prefix_hex = hex::encode(&kp.ek()[..16]);
        assert_eq!(
            prefix_hex, expected_ek_prefix_hex,
            "derivation for {alg:?} changed — this is a FROZEN cross-app vector \
             (must match app-core/src/pqkeys.rs in the graffito repo)"
        );
    }
}
