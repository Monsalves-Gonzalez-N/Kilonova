"""Logging setup for the CLIs.

Library modules only do ``logger = logging.getLogger(__name__)`` and never
configure handlers; ``setup_logging`` is called once, in each CLI ``main``.
"""

import logging


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
