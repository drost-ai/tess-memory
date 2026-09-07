from contextlib import contextmanager
from unittest import mock
import unittest

import numpy as np

from agentic_memory.embed import OpenAICompatEmbedder


class EmbeddingTransportTests(unittest.TestCase):
    def embedder(self):
        e = OpenAICompatEmbedder.__new__(OpenAICompatEmbedder)
        e._url, e._key, e._model, e._dims = 'https://example.invalid/embeddings', 'synthetic', 'test', 2
        return e

    def test_response_indices_preserve_input_order_and_guard_wraps_requests(self):
        e = self.embedder()
        entries = []
        @contextmanager
        def guard():
            entries.append('start')
            yield
            entries.append('end')
        e.request_guard = guard
        response = mock.Mock()
        response.json.return_value = {'data': [{'index': 1, 'embedding': [0, 1]}, {'index': 0, 'embedding': [1, 0]}]}
        with mock.patch('httpx.post', return_value=response):
            np.testing.assert_array_equal(e._post(['a', 'b']), np.eye(2))
        self.assertEqual(entries, ['start', 'end'])

    def test_wrong_dimensions_never_become_stored_vectors(self):
        e = self.embedder()
        response = mock.Mock()
        response.json.return_value = {'data': [{'index': 0, 'embedding': [1]}]}
        with mock.patch('httpx.post', return_value=response), mock.patch('time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'invalid dimensions'):
                e._post(['a'])


if __name__ == '__main__':
    unittest.main()
