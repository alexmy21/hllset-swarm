"""
Immutable Chinese-character HLLSets (Julia backend).
Only responsibility: load-once, read-only lookup.
"""
from __future__ import annotations
import json, gzip
from pathlib import Path
from typing import Dict
from .hllset_wrapper import HllSet

__all__ = ["get_kernel", "KERNEL"]

raw = [
{"char":"人","pinyin":"rén","defs":["由类人猿进化而成的能制造和使用工具进行劳动的高等动物"],"rad":"人","stroke":2},
{"char":"工","pinyin":"gōng","defs":["工人，从事体力劳动的人"],"rad":"工","stroke":3},
{"char":"仁","pinyin":"rén","defs":["亲也，爱人，人与人相互亲爱"],"rad":"人","stroke":4},
{"char":"信","pinyin":"xìn","defs":["诚也，诚实不欺"],"rad":"人","stroke":9},
{"char":"众","pinyin":"zhòng","defs":["多也，三人以上为众"],"rad":"人","stroke":6},
{"char":"休","pinyin":"xiū","defs":["息止也，人倚木而息"],"rad":"人","stroke":6},
{"char":"体","pinyin":"tǐ","defs":["人之总称，身体"],"rad":"人","stroke":7},
{"char":"作","pinyin":"zuò","defs":["起也，从事某种活动"],"rad":"人","stroke":7},
{"char":"使","pinyin":"shǐ","defs":["令也，支使人做事"],"rad":"人","stroke":8},
{"char":"健","pinyin":"jiàn","defs":["强也，健康有力"],"rad":"人","stroke":11},

{"char":"水","pinyin":"shuǐ","defs":["无色无味透明液体，生命之源"],"rad":"水","stroke":4},
{"char":"河","pinyin":"hé","defs":["水之道也，大川通流"],"rad":"水","stroke":8},
{"char":"江","pinyin":"jiāng","defs":["水之大者，长江之谓"],"rad":"水","stroke":6},
{"char":"海","pinyin":"hǎi","defs":["百川所归，广大水域"],"rad":"水","stroke":10},
{"char":"湖","pinyin":"hú","defs":["积水大泽，陆地中的大片水域"],"rad":"水","stroke":12},

{"char":"木","pinyin":"mù","defs":["树也，植物之总名"],"rad":"木","stroke":4},
{"char":"林","pinyin":"lín","defs":["平土有丛木曰林"],"rad":"木","stroke":8},
{"char":"森","pinyin":"sēn","defs":["木多貌，三木为森"],"rad":"木","stroke":12},
{"char":"板","pinyin":"bǎn","defs":["片木也，木板"],"rad":"木","stroke":8},
{"char":"桥","pinyin":"qiáo","defs":["水上横木，以济行者"],"rad":"木","stroke":10},

{"char":"口","pinyin":"kǒu","defs":["人所以言食也，嘴"],"rad":"口","stroke":3},
{"char":"唱","pinyin":"chàng","defs":["发歌也，口中发出乐音"],"rad":"口","stroke":11},
{"char":"和","pinyin":"hé","defs":["相应也，声音协调"],"rad":"口","stroke":8},
{"char":"品","pinyin":"pǐn","defs":["众庶也，三口为品"],"rad":"口","stroke":9},

{"char":"心","pinyin":"xīn","defs":["人心，思维器官"],"rad":"心","stroke":4},
{"char":"想","pinyin":"xiǎng","defs":["冀思也，心中有所冀"],"rad":"心","stroke":13},
{"char":"意","pinyin":"yì","defs":["志也，心中所向"],"rad":"心","stroke":13},
{"char":"愿","pinyin":"yuàn","defs":["谨也，本志所向往"],"rad":"心","stroke":14},

{"char":"日","pinyin":"rì","defs":["太阳，白昼"],"rad":"日","stroke":4},
{"char":"明","pinyin":"míng","defs":["照也，日月交辉"],"rad":"日","stroke":8},
{"char":"春","pinyin":"chūn","defs":["四时之首，万物发生"],"rad":"日","stroke":9},
{"char":"晨","pinyin":"chén","defs":["早也，日初出时"],"rad":"日","stroke":11},

{"char":"山","pinyin":"shān","defs":["土之高者，有石而高"],"rad":"山","stroke":3},
{"char":"岳","pinyin":"yuè","defs":["大山也，五岳名山"],"rad":"山","stroke":8},
{"char":"岸","pinyin":"àn","defs":["水涯而高者"],"rad":"山","stroke":8},

{"char":"土","pinyin":"tǔ","defs":["地之吐生物者也"],"rad":"土","stroke":3},
{"char":"坡","pinyin":"pō","defs":["阪也，倾斜之地"],"rad":"土","stroke":8},
{"char":"城","pinyin":"chéng","defs":["所以盛民也，城墙"],"rad":"土","stroke":9},

{"char":"火","pinyin":"huǒ","defs":["燬也，化物之气"],"rad":"火","stroke":4},
{"char":"炎","pinyin":"yán","defs":["火光上也，二火为炎"],"rad":"火","stroke":8},
{"char":"灰","pinyin":"huī","defs":["火过为灰，烬也"],"rad":"火","stroke":6},

{"char":"女","pinyin":"nǚ","defs":["妇人也，女性通称"],"rad":"女","stroke":3},
{"char":"好","pinyin":"hǎo","defs":["美也，女子为美"],"rad":"女","stroke":6},
{"char":"妈","pinyin":"mā","defs":["母也，女马合成"],"rad":"女","stroke":6},

{"char":"子","pinyin":"zǐ","defs":["十一月阳气动，万物滋，人以为偁"],"rad":"子","stroke":3},
{"char":"学","pinyin":"xué","defs":["觉悟也，受教传知"],"rad":"子","stroke":8},
{"char":"字","pinyin":"zì","defs":["乳也，屋下生子，文字"],"rad":"子","stroke":6},

{"char":"手","pinyin":"shǒu","defs":["拳也，人体上肢"],"rad":"手","stroke":4},
{"char":"打","pinyin":"dǎ","defs":["击也，手相敲击"],"rad":"手","stroke":5},
{"char":"拍","pinyin":"pāi","defs":["拊也，轻拍"],"rad":"手","stroke":8},

{"char":"目","pinyin":"mù","defs":["人眼也，视觉器官"],"rad":"目","stroke":5},
{"char":"看","pinyin":"kàn","defs":["睎也，目之所向"],"rad":"目","stroke":9},
{"char":"睛","pinyin":"jīng","defs":["目珠子也，眼球"],"rad":"目","stroke":13},

{"char":"言","pinyin":"yán","defs":["直言曰言，说话"],"rad":"言","stroke":7},
{"char":"信","pinyin":"xìn","defs":["诚也，人言为信"],"rad":"言","stroke":9},
{"char":"讲","pinyin":"jiǎng","defs":["和解也，言语阐述"],"rad":"言","stroke":6}
]

class Kernel:
    """Read-only container of Julia HLLSets (one per character)."""
    __slots__ = ("_data",)

    def __init__(self, path: str | Path = None):

        if path is not None:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data_source: dict[str, dict] = json.load(f)          # {"人": {"rad": "人", "defs": [...] , ...} , ...}
        else:
            data_source = raw                             # use embedded data

        self._data: Dict[str, jl.HllSet] = {}
        for rec in data_source:
            c = rec["char"]
            h = HllSet(P=10)          # 1024 registers
            h.add(rec["rad"])
            for d in rec["defs"]:
                h.add(d)
            h.add(str(rec["stroke"]))
            self._data[c] = h

    def __getitem__(self, ch: str) -> jl.HllSet:
        return self._data[ch]

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

# singleton loaded once at import
_KERNEL_PATH = None

_KERNEL_PATH = Path(__file__).with_name("80k_ccd.json.gz")

if not _KERNEL_PATH.exists():
    _KERNEL_PATH = None

print("Loading HLLSet kernel from:", _KERNEL_PATH)

KERNEL = Kernel(_KERNEL_PATH)   # public singleton

def get_kernel() -> Kernel:
    return KERNEL