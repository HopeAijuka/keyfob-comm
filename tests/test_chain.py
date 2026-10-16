"""Self-checks of every block.  Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions.config import Config
from functions.framing import assemble_packet, crc16_ccitt, frame_layout, parse_packet
from functions.line_coding import manchester_decode, manchester_encode
from functions.link import run_link
from functions.metrics import ber_ook_noncoherent, mse_from_ber, sqnr_theory_db
from functions.pcm import pcm_decode, pcm_encode
from functions.quantizer import dequantize, quantize
from functions.rolling_code import (CMD_UNLOCK, KeyFob, VehicleReceiver, keeloq_decrypt,
                                    keeloq_encrypt)
from functions.transmitter import transmit

CFG = Config()


class TestBlocks(unittest.TestCase):
    def test_crc_check_value(self):
        self.assertEqual(int(crc16_ccitt(np.frombuffer(b"123456789", dtype=np.uint8))), 0x29B1)

    def test_pcm_roundtrip(self):
        idx = np.arange(256)
        np.testing.assert_array_equal(pcm_decode(pcm_encode(idx, 8), 8), idx)

    def test_quantizer_error_bound(self):
        x = np.linspace(-1, 1, 5001)
        idx, xq, delta = quantize(x, 8)
        self.assertLessEqual(np.abs(x - xq).max(), delta / 2 + 1e-12)
        np.testing.assert_allclose(dequantize(idx, 8), xq)

    def test_manchester_roundtrip(self):
        bits = np.random.default_rng(0).integers(0, 2, 500).astype(np.uint8)
        chips = manchester_encode(bits)
        np.testing.assert_array_equal(manchester_decode(chips.astype(float)), bits)
        self.assertAlmostEqual(chips.mean(), 0.5)                      # 50 % duty by construction
        self.assertTrue(np.all(chips[0::2] + chips[1::2] == 1))

    def test_keeloq_roundtrip(self):
        for blk in (0, 1, 0xDEADBEEF, 0xFFFFFFFF):
            self.assertEqual(keeloq_decrypt(keeloq_encrypt(blk, CFG.fob_key), CFG.fob_key), blk)

    def test_packet_parse(self):
        tx = transmit(CFG, KeyFob(CFG.device_id, CFG.fob_key), CMD_UNLOCK)
        p = parse_packet(tx.packet_bits[None, :], CFG.n_payload_bits, CFG.sync_word)
        self.assertTrue(p["crc_ok"][0])
        self.assertEqual(int(p["device_id"][0]), CFG.device_id)
        self.assertEqual(tx.packet_bits.size, frame_layout(CFG.n_payload_bits)["total"][1])

    def test_theory_helpers(self):
        self.assertAlmostEqual(sqnr_theory_db(8), 49.92, places=2)
        self.assertAlmostEqual(float(ber_ook_noncoherent(0.0)), 0.5 * np.exp(-0.5), places=9)
        tx = transmit(CFG, KeyFob(CFG.device_id, CFG.fob_key), CMD_UNLOCK)
        q = tx.x_s - tx.x_q
        self.assertAlmostEqual(mse_from_ber(0.0, q, tx.pcm_bits, 8)[0], np.mean(q ** 2), places=12)
        self.assertAlmostEqual(mse_from_ber(0.5, q, tx.pcm_bits, 8)[0], 0.5 + 1 / 3, delta=0.01)


class TestLink(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.tx = transmit(CFG, KeyFob(CFG.device_id, CFG.fob_key), CMD_UNLOCK)

    def test_clean_channel_is_lossless(self):
        _, leads, _, res, _ = run_link(self.tx.burst, self.tx.burst_q, CFG, 30.0, self.rng, n=5)
        self.assertTrue(res.accepted.all())
        np.testing.assert_array_equal(res.bits, np.tile(self.tx.packet_bits, (5, 1)))
        self.assertTrue((np.abs(res.t0 - leads) <= 1).all())

    def test_noise_destroys_packet(self):
        _, _, _, res, _ = run_link(self.tx.burst, self.tx.burst_q, CFG, -8.0, self.rng, n=5)
        self.assertFalse(res.accepted.any())

    def test_wrong_id_is_ignored(self):
        _, _, _, res, _ = run_link(self.tx.burst, self.tx.burst_q, CFG, 20.0, self.rng, n=1,
                                   expected_id=CFG.other_device_id)
        self.assertTrue(res.crc_ok[0] and not res.id_ok[0] and not res.accepted[0])


class TestRollingCode(unittest.TestCase):
    def test_replay_and_window(self):
        fob, car = KeyFob(CFG.device_id, CFG.fob_key), VehicleReceiver(16)
        car.learn(CFG.device_id, CFG.fob_key)
        cmd, hop, _ = fob.press(CMD_UNLOCK)
        self.assertTrue(car.process(CFG.device_id, cmd, hop).accepted)
        self.assertFalse(car.process(CFG.device_id, cmd, hop).accepted)          # replay
        fob.press_out_of_range(3)
        cmd, hop, _ = fob.press(CMD_UNLOCK)
        self.assertTrue(car.process(CFG.device_id, cmd, hop).accepted)           # inside window
        fob.press_out_of_range(30)
        cmd, hop, _ = fob.press(CMD_UNLOCK)
        self.assertFalse(car.process(CFG.device_id, cmd, hop).accepted)          # outside window
        cmd, hop, _ = fob.press(CMD_UNLOCK)
        self.assertTrue(car.process(CFG.device_id, cmd, hop).accepted)           # re-sync

    def test_forged_hop_rejected(self):
        car = VehicleReceiver(16)
        car.learn(CFG.device_id, CFG.fob_key)
        self.assertFalse(car.process(CFG.device_id, CMD_UNLOCK, 0x12345678).accepted)


if __name__ == "__main__":
    unittest.main()
