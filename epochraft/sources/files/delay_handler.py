from __future__ import annotations

import logging
import os
import threading
import time
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


DEFAULT_DELAY = 1.0


class ProtocolDelayHandler:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.protocols: dict[str, tuple[float, threading.Lock]] = {}  # (last_time, lock)
        self.delay: dict[str, float] = {}

    @staticmethod
    def _get_env_name(protocol: str) -> str:
        return f"EPOCHRAFT_OPEN_DELAY_{protocol}"

    def _get_delay_for_protocol(self, protocol: str) -> float:
        if protocol in self.delay:
            return self.delay[protocol]

        env_name = self._get_env_name(protocol)
        delay_str = os.environ.get(env_name, None)

        if delay_str is None:
            if protocol == "FILE":
                delay = 0.0
            else:
                delay = DEFAULT_DELAY

                logger.debug(
                    f"Using a default delay of {delay} seconds for protocol {protocol}. "
                    f"We don't open {protocol} file more than once per {delay} second. "
                    f"This can be configured using the environment variable "
                    f"`EPOCHRAFT_OPEN_DELAY_{protocol}`."
                )
        else:
            try:
                assert delay_str is not None
                delay = float(delay_str)
            except ValueError:
                logger.error(
                    f"Invalid delay value from environment variable: {env_name} = {delay_str}"
                )
                delay = DEFAULT_DELAY if protocol != "FILE" else 0.0

        self.delay[protocol] = delay
        return delay

    def __call__(self, url: str) -> None:
        # Extract the protocol/scheme from the URL
        parsed_url = urlparse(url)
        protocol = parsed_url.scheme.upper() if parsed_url.scheme else "FILE"

        # Find the corresponding environment variable for the delay
        delay = self._get_delay_for_protocol(protocol)

        lock = self._get_protocol_lock(protocol)
        with lock:
            last_time = self.protocols[protocol][0]
            elapsed_time = time.time() - last_time
            if elapsed_time < delay:
                logger.debug(
                    f"Waiting for {delay - elapsed_time:.2f} seconds before opening {url}."
                )
                time.sleep(delay - elapsed_time)
            self.protocols[protocol] = (time.time(), lock)

    def _get_protocol_lock(self, protocol: str) -> threading.Lock:
        with self.lock:
            if protocol not in self.protocols:
                with threading.Lock():
                    if protocol not in self.protocols:
                        self.protocols[protocol] = (0, threading.Lock())
            return self.protocols[protocol][1]
