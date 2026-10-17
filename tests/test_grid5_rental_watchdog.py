"""No provider calls: ownership and cost bounds guard the exact task rental."""

import copy
import json
from types import SimpleNamespace
import unittest

from probity_benchmark import grid5_rental_watchdog as watchdog


def receipt():
    return {'instance_id': 123, 'label': 'probity-iteration5-20261016', 'existing_instance_ids': [99],
            'created_at_unix': 100, 'destroy_at_unix': 32500, 'max_runtime_hours': 9,
            'cleanup_allowance_hours': .25, 'hourly_upper_usd': 1.53, 'traffic_reserve_usd': .8,
            'reservation_usd': 15}


class RentalTests(unittest.TestCase):
    def test_runtime_reserve_rate_and_ownership_gates(self):
        watchdog.validate_receipt(receipt())
        for key, value in [('max_runtime_hours', 10), ('hourly_upper_usd', 1.61), ('reservation_usd', 14),
                           ('destroy_at_unix', 33000), ('existing_instance_ids', [123]), ('label', 'unrelated')]:
            altered = {**receipt(), key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                watchdog.validate_receipt(altered)

    def test_predeadline_never_calls_provider(self):
        with self.assertRaisesRegex(ValueError, 'deadline'):
            watchdog.destroy_once(receipt(), 200, run=lambda *a, **k: self.fail('provider called'))

    def runner(self, inventories):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            value = inventories.pop(0) if args[1] == 'show' else {'success': True}
            return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr='')
        return calls, run

    def test_unrelated_label_and_malformed_inventory_never_destroy(self):
        for inventory in ([{'id': 123, 'label': 'another-task'}], {'unexpected': []}):
            calls, run = self.runner([inventory])
            with self.assertRaises(ValueError):
                watchdog.destroy_once(receipt(), 32501, run=run)
            self.assertEqual(len(calls), 1)

    def test_zero_returncode_requires_independently_verified_absence(self):
        own = {'id': 123, 'label': receipt()['label']}
        unrelated = {'id': 99, 'label': 'leave-me'}
        calls, run = self.runner([[own, unrelated], [own, unrelated]])
        self.assertFalse(watchdog.destroy_once(receipt(), 32501, run=run)['verified_absent'])
        self.assertEqual(calls[1], ['vastai', 'destroy', 'instance', '123', '--yes', '--raw'])
        calls, run = self.runner([[own, unrelated], [unrelated]])
        self.assertTrue(watchdog.destroy_once(receipt(), 32501, run=run)['verified_absent'])
        calls, run = self.runner([[unrelated]])
        result = watchdog.destroy_once(receipt(), 32501, run=run)
        self.assertTrue(result['verified_absent'])
        self.assertFalse(result['destroy_requested'])


if __name__ == '__main__':
    unittest.main()
