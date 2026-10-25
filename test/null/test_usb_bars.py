import struct, unittest
from tinygrad.runtime.support.system import System

class FakeASM24:
  # config space only: the gpu on bus 4 has a resizable bar capability at 0x100 with two entries
  def __init__(self, vendor:int, entries:list[tuple[int, int]]):
    self.cfg: dict[int, bytearray] = {}
    gpu = self._bus(4)
    struct.pack_into('<H', gpu, 0, vendor)
    struct.pack_into('<I', gpu, 0x100, 0x15 | (1 << 16)) # rebar, version 1, no next cap
    for i, (bar, cap) in enumerate(entries):
      struct.pack_into('<II', gpu, 0x104 + 8 * i, cap, bar | ((len(entries) << 5) if i == 0 else 0))
  def _bus(self, bus:int) -> bytearray: return self.cfg.setdefault(bus, bytearray(0x1000))
  def pcie_cfg_req(self, byte_addr:int, bus:int=1, dev:int=0, fn:int=0, value:int|None=None, size:int=4):
    fmt = {1: '<B', 2: '<H', 4: '<I'}[size]
    if value is None: return struct.unpack_from(fmt, self._bus(bus), byte_addr)[0]
    struct.pack_into(fmt, self._bus(bus), byte_addr, value)

def size_field(usb:FakeASM24, ent:int) -> int: return (struct.unpack_from('<I', usb.cfg[4], 0x108 + 8 * ent)[0] >> 8) & 0x1f

class TestUSBBars(unittest.TestCase):
  def setup(self, usb, vendor=None): System.pci_setup_usb_bars(usb, gpu_bus=4, mem_base=0x10000000, pref_mem_base=32 << 30, vendor=vendor)  # type: ignore

  def test_nvidia_resizes_bar1(self):
    usb = FakeASM24(0x10de, [(1, 1 << 17), (3, 1 << 5)]) # cap bit 4+n is 1mb << n: bar1 up to 8gb, bar3 up to 2mb
    self.setup(usb, vendor=0x10de)
    self.assertEqual(size_field(usb, 0), 13) # 8gb = 1mb << 13
    self.assertEqual(size_field(usb, 1), 0)

  def test_amd_resizes_bar0(self):
    usb = FakeASM24(0x1002, [(0, 1 << 18)])
    self.setup(usb)
    self.assertEqual(size_field(usb, 0), 14)

  def test_wrong_vendor_keeps_bars(self):
    usb = FakeASM24(0x1002, [(0, 1 << 14)])
    with self.assertRaises(RuntimeError): self.setup(usb, vendor=0x10de)
    self.assertEqual(size_field(usb, 0), 0)

if __name__ == "__main__":
  unittest.main()
