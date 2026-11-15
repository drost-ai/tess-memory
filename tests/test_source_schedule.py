from pathlib import Path
import unittest

from tess_memory.source_schedule import source_scan_order


def source(name, modified=0):
    return (modified * 1e9, None, Path('/sources') / name, 'project', None)


class SourceScheduleTests(unittest.TestCase):
    def order(self, found, *, limit=4, cursor=None, known=()):
        return source_scan_order(found, limit=limit, cursor=cursor, known_paths=set(known), now=10000)

    def test_recent_priority_cannot_consume_the_entire_read_budget(self):
        found = [source('old-a'), source('old-b')] + [source(f'recent-{i}', 9999-i) for i in range(6)]
        selected = self.order(found)[:4]
        self.assertEqual([r[0][2].name for r in selected], ['recent-0', 'recent-1', 'old-a', 'old-b'])
        self.assertEqual([r[1] for r in selected], [False, False, True, True])

    def test_initial_scan_starts_at_unseen_work_and_cursor_wraps(self):
        found = [source('a'), source('b'), source('c')]
        first = self.order(found, limit=1, known=('/sources/a',))
        self.assertEqual(first[0][0][2].name, 'b')
        second = self.order(found, limit=1, cursor='/sources/b')
        self.assertEqual(second[0][0][2].name, 'c')
        third = self.order(found, limit=1, cursor='/sources/c')
        self.assertEqual(third[0][0][2].name, 'a')

    def test_removed_cursor_path_and_duplicate_adapter_entries_do_not_drop_sources(self):
        a, b, c = source('a'), source('b'), source('c')
        ordered = self.order([a, a, b, c], limit=1, cursor='/sources/bb')
        self.assertEqual([r[0][2].name for r in ordered], ['c', 'a', 'a', 'b'])

    def test_unbounded_scan_retains_newest_first_order_without_cursor_writes(self):
        ordered = self.order([source('old', 1), source('new', 2)], limit=None, cursor='/sources/old')
        self.assertEqual([r[0][2].name for r in ordered], ['new', 'old'])
        self.assertTrue(all(not r[1] for r in ordered))


if __name__ == '__main__':
    unittest.main()
