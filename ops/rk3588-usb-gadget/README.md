# H96 Max V58 USB ACM/NCM Gadget

This configuration turns the H96 Max V58 black USB 2.0 Maskrom port into a
Linux USB peripheral after boot. The Raspberry Pi host should enumerate the
device as `1d6b:0104` and create `/dev/ttyACM0` plus `usb0`; the RK3588 side
exposes `/dev/ttyGS0` with an autologin getty and uses `169.254.4.1/16` for
the point-to-point USB network.

The kernel already contains DWC3 dual-role, ConfigFS, libcomposite and ACM
support. The deployment therefore changes only:

- `usb_host0_xhci` from `dr_mode = "host"` to `dr_mode = "peripheral"`;
- the DTB stored in the boot filesystem;
- two systemd units and the ConfigFS helper stored in the root filesystem.

Keep the original `images/boot.ext4` and `images/rootfs.ext4` unchanged. Build
new `boot-usb-acm.ext4` and `rootfs-usb-acm.ext4` copies, verify them offline,
then write only the existing `boot` and `rootfs` GPT partitions. Do not erase
the flash or rewrite the SPL/U-Boot area.

During recovery, verify ACM serial first so USB role failures remain separate
from composite-function failures. After ACM works, install
`rk3588-usb-gadget` as `/usr/local/sbin/rk3588-usb-gadget` and install
`20-rk3588-usb-gadget.conf` in `/etc/NetworkManager/conf.d/`. The latter
overrides Ubuntu's globally unmanaged Ethernet policy for `usb0`, allowing the
saved `rk3588-usb` NetworkManager connection to configure the static address.

The stock Android OTG procedure commonly uses `dr_mode = "otg"` because the
Android USB framework selects device mode at runtime. This headless Ubuntu
image has no equivalent role manager, so the dedicated Raspberry Pi debug link
uses fixed `peripheral` mode to make the DWC3 UDC available to ConfigFS.
