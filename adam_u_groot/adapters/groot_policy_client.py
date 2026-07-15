"""Lightweight ZMQ client for the Isaac-GR00T policy server.

This avoids importing the full ``gr00t`` package in the Isaac Lab conda env.
The GR00T model server typically runs in the same conda env as Isaac
(``adam-u-groot-unified``) or a separate ``lerobot-groot`` env; this client only
sends observations and receives actions over the wire.
"""

from __future__ import annotations

import functools
import io
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class _MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(_MsgSerializer._safe_encode, chain=_MsgSerializer._encode_custom)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(_MsgSerializer._safe_decode, chain=_MsgSerializer._decode_custom)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj, chain=None):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                "convert to a numeric dtype before sending."
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:
                payload = obj.get("as_npy", obj.get(b"as_npy"))
                if payload is None:
                    raise ValueError("Malformed ndarray payload: marker present but 'as_npy' missing")
                return np.load(io.BytesIO(payload), allow_pickle=False)
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload.")
        return mnp.decode(obj, chain=chain)

    @staticmethod
    def _encode_custom(obj):
        return obj

    @staticmethod
    def _decode_custom(obj):
        return obj


class GrootPolicyClient:
    """Minimal client compatible with ``gr00t.policy.server_client.PolicyServer``."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._closed = False
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            self._call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def _call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(_MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise

        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self._call_endpoint(
            "get_action",
            {"observation": observation, "options": options},
        )
        return tuple(response)
