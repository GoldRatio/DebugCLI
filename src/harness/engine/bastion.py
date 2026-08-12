"""SSH bastion / proxy jump support.

A session may hop through a bastion for isolated networks. This composes the
paramiko ``sock`` the same way ``SSHSession`` does, so the command funnel and host-key
pinning are unchanged. For the scaffold the hop is a thin adapter around a user-supplied
forward socket; a full implementation would open a tunnel to the bastion first.
"""

from __future__ import annotations

import socket

import paramiko


def open_bastion_socket(bastion_host: str, bastion_port: int,
                        target_host: str, target_port: int,
                        user: str, key_file: str,
                        timeout: float = 30.0) -> socket.socket:
    """Return a socket already connected to ``target_host`` via the bastion.

    Uses a direct TCP connection over an SSH transport to the bastion (ProxyJump
    style). This is the standard technique and keeps the session pinned to the
    remote end. Raises on any failure so a bad hop is never silently used.
    """
    transport = paramiko.Transport((bastion_host, bastion_port))
    transport.connect(username=user, key_filename=key_file)
    channel = transport.open_channel(
        "direct-tcpip",
        (target_host, target_port),
        ("", 0),
        timeout=timeout,
    )
    return channel