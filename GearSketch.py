# -*- coding: utf-8 -*-
"""
GearSketch - Autodesk Fusion アドイン (ラック)
スケッチ編集中に「作成」メニューの「ラック (モジュール指定)」から、
選んだ2点(または1本の線)の間に、任意モジュールのラック歯形をスケッチします。

  - 2点を結ぶ線 = 基準線 (ピッチ線 / 歯底線 / 歯先線 から選択)
  - モジュール、圧力角、歯先・歯元の係数、歯元R・歯先R、歯厚減少量(バックラッシ)
  - 始点を歯溝の中心 / 歯の中心に合わせる + 位相オフセット
  - 閉じた輪郭(背面付き)にすれば、そのまま押し出し可能
  - 端は指定した2点の位置で歯形を正確に切ります

単位: API 内部は cm。
"""

import json
import math
import traceback

import adsk.core
import adsk.fusion

# ==== GEOMETRY BEGIN ====
EPS = 1e-9

REF_PITCH = 'ピッチ線'
REF_ROOT = '歯底線'
REF_TIP = '歯先線'
START_GAP = '歯溝の中心'
START_TOOTH = '歯の中心'
MODE_PARAM = 'パラメトリック (まとめて動く)'
MODE_FIXED = '固定形状 (両端で正確に切る)'


class RackError(Exception):
    pass


def rack_dims(prm):
    m = prm['module']
    t = math.tan(prm['pa'])
    d = {
        'm': m, 't': t,
        'p': math.pi * m,
        'ha': prm['ha'] * m,
        'hf': prm['hf'] * m,
        'rr': prm['root_r'] * m,
        'rt': prm.get('tip_r', 0.0) * m,
    }
    d['s'] = d['p'] / 2 - prm.get('thin', 0.0)
    d['tip_w'] = d['s'] - 2 * d['ha'] * t
    d['gap_w'] = d['p'] - d['s'] - 2 * d['hf'] * t
    return d


def rack_vertices(L, prm, d):
    """ピッチ線を y=0 とした折れ線頂点 (x, y, 角R) を x 増加順に返す。"""
    p, s, t, ha, hf = d['p'], d['s'], d['t'], d['ha'], d['hf']
    x0 = (0.0 if prm['start'] == START_TOOTH else p / 2) + prm['phase']
    k0 = int(math.floor((-2 * p - x0) / p))
    k1 = int(math.ceil((L + 2 * p - x0) / p))
    verts = []
    centers = []
    for k in range(k0, k1 + 1):
        xc = x0 + k * p
        centers.append(xc)
        verts.append((xc - s / 2 - hf * t, -hf, d['rr']))
        verts.append((xc - s / 2 + ha * t, ha, d['rt']))
        verts.append((xc + s / 2 - ha * t, ha, d['rt']))
        verts.append((xc + s / 2 + hf * t, -hf, d['rr']))
    return verts, centers


def fillet_path(verts):
    """角R付きの折れ線を ('L', a, b) / ('A', c, r, a0, a1) の列に変換。"""
    prims = []
    cur = (verts[0][0], verts[0][1])
    used_prev = 0.0
    for i in range(1, len(verts) - 1):
        A, P, B = verts[i - 1], verts[i], verts[i + 1]
        r = P[2]
        seg = math.hypot(P[0] - A[0], P[1] - A[1])
        if r <= EPS:
            prims.append(('L', cur, (P[0], P[1])))
            cur, used_prev = (P[0], P[1]), 0.0
            continue
        d1 = ((P[0] - A[0]) / seg, (P[1] - A[1]) / seg)
        lb = math.hypot(B[0] - P[0], B[1] - P[1])
        d2 = ((B[0] - P[0]) / lb, (B[1] - P[1]) / lb)
        cr = d1[0] * d2[1] - d1[1] * d2[0]
        dt = d1[0] * d2[0] + d1[1] * d2[1]
        phi = math.atan2(abs(cr), dt)
        tl = r * math.tan(phi / 2)
        if seg - used_prev < tl - 1e-12:
            raise RackError('R が大きすぎて歯形に収まりません。歯元R・歯先Rの係数を小さくしてください。')
        sgn = 1.0 if cr > 0 else -1.0
        T1 = (P[0] - d1[0] * tl, P[1] - d1[1] * tl)
        T2 = (P[0] + d2[0] * tl, P[1] + d2[1] * tl)
        C = (T1[0] - d1[1] * sgn * r, T1[1] + d1[0] * sgn * r)
        a0 = math.atan2(T1[1] - C[1], T1[0] - C[0])
        prims.append(('L', cur, T1))
        prims.append(('A', C, r, a0, a0 + sgn * phi))
        cur, used_prev = T2, tl
    last = verts[-1]
    prims.append(('L', cur, (last[0], last[1])))
    return [q for q in prims if prim_len(q) > 1e-9]


def prim_len(q):
    if q[0] == 'L':
        return math.hypot(q[2][0] - q[1][0], q[2][1] - q[1][1])
    return abs(q[4] - q[3]) * q[2]


def arc_pt(C, r, a):
    return (C[0] + r * math.cos(a), C[1] + r * math.sin(a))


def prim_start(q):
    return q[1] if q[0] == 'L' else arc_pt(q[1], q[2], q[3])


def prim_end(q):
    return q[2] if q[0] == 'L' else arc_pt(q[1], q[2], q[4])


def _arc_angle_at_x(C, r, a0, a1, x):
    c = max(-1.0, min(1.0, (x - C[0]) / r))
    th = math.acos(c)
    lo, hi = min(a0, a1), max(a0, a1)
    for k in range(-3, 4):
        for b in (th, -th):
            cand = b + 2 * math.pi * k
            if lo - 1e-9 <= cand <= hi + 1e-9:
                return cand
    raise RackError('円弧の切断に失敗しました')


def clip_path(prims, x0, x1):
    """x 方向に単調な要素列を [x0, x1] で切り取る。"""
    out = []
    for q in prims:
        xa, xb = prim_start(q)[0], prim_end(q)[0]
        if xb <= x0 + 1e-12 or xa >= x1 - 1e-12:
            continue
        if q[0] == 'L':
            (ax, ay), (bx, by) = q[1], q[2]

            def at(x):
                return (x, ay + (x - ax) / (bx - ax) * (by - ay))
            a = q[1] if xa >= x0 else at(x0)
            b = q[2] if xb <= x1 else at(x1)
            nq = ('L', a, b)
        else:
            C, r, a0, a1 = q[1], q[2], q[3], q[4]
            na0 = a0 if xa >= x0 else _arc_angle_at_x(C, r, a0, a1, x0)
            na1 = a1 if xb <= x1 else _arc_angle_at_x(C, r, a0, a1, x1)
            nq = ('A', C, r, na0, na1)
        if prim_len(nq) > 1e-9:
            out.append(nq)
    return out


def shift_y(prims, dy):
    out = []
    for q in prims:
        if q[0] == 'L':
            out.append(('L', (q[1][0], q[1][1] + dy), (q[2][0], q[2][1] + dy)))
        else:
            out.append(('A', (q[1][0], q[1][1] + dy), q[2], q[3], q[4]))
    return out


def build_profile(L, prm):
    """基準線(始点→終点)を x 軸とした局所座標でのラック歯形要素列と情報を返す。"""
    d = checked_dims(prm)
    verts, centers = rack_vertices(L, prm, d)
    prims = clip_path(fillet_path(verts), 0.0, L)
    y_pitch = {REF_PITCH: 0.0, REF_ROOT: d['hf'], REF_TIP: -d['ha']}[prm['ref']]
    prims = shift_y(prims, y_pitch)
    teeth = sum(1 for c in centers if 0.0 <= c <= L)
    return prims, d, y_pitch, teeth


def checked_dims(prm):
    d = rack_dims(prm)
    if prm['module'] <= 0:
        raise RackError('モジュールは正の値にしてください')
    if prm['pa'] < math.radians(0.5) or prm['pa'] > math.radians(45):
        raise RackError('圧力角は 0.5°〜45° の範囲にしてください')
    if d['tip_w'] <= 1e-9:
        raise RackError('歯先の幅がなくなります (歯先係数か圧力角か歯厚減少量が大きすぎます)')
    if d['gap_w'] <= 1e-9:
        raise RackError('歯溝の幅がなくなります (歯元係数か圧力角が大きすぎます)')
    return d


class _Chain:
    """要素を端点共有でつないで描く。"""

    def __init__(self, sketch, to_sk, start):
        self.sk = sketch
        self.to_sk = to_sk
        self.cur = start          # SketchPoint か Point3D
        self.first = None
        self.curves = []

    def _p(self, xy):
        return self.to_sk(xy[0], xy[1])

    def line(self, end):
        """end は (x, y) か SketchPoint。"""
        e = end if isinstance(end, adsk.fusion.SketchPoint) else self._p(end)
        ln = self.sk.sketchCurves.sketchLines.addByTwoPoints(self.cur, e)
        self.curves.append(ln)
        if self.first is None:
            self.first = ln.startSketchPoint
        self.cur = ln.endSketchPoint
        return ln

    def arc(self, mid, end):
        endp = self._p(end)
        a = self.sk.sketchCurves.sketchArcs.addByThreePoints(self.cur, self._p(mid), endp)
        self.curves.append(a)
        s, e = a.startSketchPoint, a.endSketchPoint
        if self.first is None:
            self.first = e if s.geometry.distanceTo(endp) < e.geometry.distanceTo(endp) else s
        self.cur = s if s.geometry.distanceTo(endp) < e.geometry.distanceTo(endp) else e
        return a

    def prim(self, q):
        if q[0] == 'L':
            self.line(q[2])
        else:
            C, r, a0, a1 = q[1], q[2], q[3], q[4]
            self.arc(arc_pt(C, r, (a0 + a1) / 2), arc_pt(C, r, a1))


def draw_rack(sketch, S, E, prm, s_sp=None, e_sp=None):
    """S, E: スケッチ空間の Point3D。s_sp, e_sp: 同じスケッチ内の SketchPoint なら接続に使う。
    戻り値: (作成カーブのリスト, 情報 dict)"""
    dx, dy = E.x - S.x, E.y - S.y
    L = math.hypot(dx, dy)
    if L < 1e-6:
        raise RackError('始点と終点が同じ位置です')
    ux, uy = dx / L, dy / L
    side = -1.0 if prm['flip'] else 1.0
    nx, ny = -uy * side, ux * side

    def to_sk(x, y):
        return adsk.core.Point3D.create(S.x + ux * x + nx * y, S.y + uy * x + ny * y, 0.0)

    prims, d, y_pitch, teeth = build_profile(L, prm)
    if not prims:
        raise RackError('歯形を作れませんでした')
    p_start, p_end = prim_start(prims[0]), prim_end(prims[-1])
    tol = 1e-6
    s_ref = s_sp if s_sp is not None else S
    e_ref = e_sp if e_sp is not None else E

    was_deferred = sketch.isComputeDeferred
    sketch.isComputeDeferred = True
    try:
        if prm['closed']:
            y_back = y_pitch - prm['back']
            if y_back >= y_pitch - d['hf'] - 1e-6:
                raise RackError('背面の位置は歯底より下(ピッチ線から歯元のたけより大きい距離)にしてください')
            ch = _Chain(sketch, to_sk, to_sk(0.0, y_back))
            if y_back + tol < 0.0 < p_start[1] - tol:
                ch.line(s_ref)
            ch.line(p_start)
            for q in prims:
                ch.prim(q)
            if y_back + tol < 0.0 < p_end[1] - tol:
                ch.line(e_ref)
            ch.line((L, y_back))
            ch.line(ch.first)
        else:
            if abs(p_start[1]) > tol:
                ch = _Chain(sketch, to_sk, s_ref)
                ch.line(p_start)
            else:
                ch = _Chain(sketch, to_sk, s_ref)
            for q in prims:
                ch.prim(q)
            if abs(p_end[1]) > tol:
                ch.line(e_ref)
            elif e_sp is not None:
                try:
                    e_sp.merge(ch.cur)
                except Exception:
                    pass
        if prm.get('fix'):
            for c in ch.curves:
                c.isFixed = True
    finally:
        sketch.isComputeDeferred = was_deferred

    info = {'pitch': d['p'], 'teeth': teeth, 'length': L, 'ha': d['ha'], 'hf': d['hf']}
    return ch.curves, info


# ---------------------------------------------------------------------------
# パラメトリック版: 歯1枚を拘束で定義 → ピッチ線方向の関連付きパターン
#   ・ピッチ線(補助線)に全体が拘束されるので、線を動かす/歯をドラッグすると一体で動く
#   ・歯数はユーザーパラメータ RackN_N で変更できる
#   ・モジュール等の変更は、ラックを選んでコマンドを再実行 → 同じピッチ線上に作り直す
# ---------------------------------------------------------------------------

ATTR_GROUP = 'GearSketch'
ATTR_RACK = 'rack'       # ピッチ線に付ける (値 = JSON パラメータ)
ATTR_MEMBER = 'member'   # ラックを構成する要素に付ける (値 = プレフィックス)


def _next_prefix(des):
    used = set()
    for a in des.findAttributes(ATTR_GROUP, ATTR_RACK):
        try:
            used.add(json.loads(a.value).get('prefix'))
        except Exception:
            pass
    i = 1
    while 'Rack{}'.format(i) in used or des.userParameters.itemByName('Rack{}_N'.format(i)):
        i += 1
    return 'Rack{}'.format(i)


def _num(v):
    return '{:.8g}'.format(v)


def _mm(v_cm):
    return _num(v_cm * 10) + ' mm'


def _deg(v_rad):
    return _num(math.degrees(v_rad)) + ' deg'


def _tag(ent, pre):
    try:
        ent.attributes.add(ATTR_GROUP, ATTR_MEMBER, pre)
    except Exception:
        pass


def _attr(ent, name):
    try:
        a = ent.attributes.itemByName(ATTR_GROUP, name)
        return a.value if a else None
    except Exception:
        return None


def find_rack(entity):
    """選択要素が既存ラックの一部(またはそのピッチ線)なら (sketch, PL, prefix, params) を返す。"""
    sk = entity.parentSketch
    pre = None
    v = _attr(entity, ATTR_RACK)
    if v:
        return sk, entity, json.loads(v)['prefix'], json.loads(v)
    pre = _attr(entity, ATTR_MEMBER)
    if pre is None:  # パターンで増えた要素
        for i in range(sk.geometricConstraints.count):
            g = sk.geometricConstraints.item(i)
            if g.objectType != adsk.fusion.RectangularPatternConstraint.classType():
                continue
            try:
                ents = list(g.entities)
                if not ents or _attr(ents[0], ATTR_MEMBER) is None:
                    continue
                if any(c == entity for c in g.createdEntities):
                    pre = _attr(ents[0], ATTR_MEMBER)
                    break
            except Exception:
                continue
    if pre is None:
        return None
    for ln in sk.sketchCurves.sketchLines:
        v = _attr(ln, ATTR_RACK)
        if v and json.loads(v).get('prefix') == pre:
            return sk, ln, pre, json.loads(v)
    return None


def delete_rack(sketch, pre):
    """ピッチ線を残してラック (歯・背面・パターン・寸法・歯数パラメータ) を削除する。"""
    des = adsk.fusion.Design.cast(sketch.parentComponent.parentDesign)
    members = []
    for i in range(sketch.geometricConstraints.count - 1, -1, -1):
        g = sketch.geometricConstraints.item(i)
        if g.objectType != adsk.fusion.RectangularPatternConstraint.classType():
            continue
        ents = list(g.entities)
        if ents and _attr(ents[0], ATTR_MEMBER) == pre:
            members.extend(list(g.createdEntities))
            g.deleteMe()
    for a in des.findAttributes(ATTR_GROUP, ATTR_MEMBER):
        if a.value == pre and a.parent is not None:
            members.append(a.parent)
    for e in members:
        try:
            if e.isValid:
                e.deleteMe()
        except Exception:
            pass
    up = des.userParameters.itemByName(pre + '_N')
    if up:
        try:
            up.deleteMe()
        except Exception:
            pass


def draw_rack_param(sketch, PL, prm, prefix=None, dry_run=False):
    """PL (SketchLine, 補助線) を基準線として拘束付きラックを作る。
    dry_run=True なら入力チェックと寸法計算だけ行い、スケッチは変更しない。"""
    VI = adsk.core.ValueInput.createByString
    AL = adsk.fusion.DimensionOrientations.AlignedDimensionOrientation
    des = adsk.fusion.Design.cast(sketch.parentComponent.parentDesign)
    d = checked_dims(prm)
    gc = sketch.geometricConstraints
    dims = sketch.sketchDimensions
    lines = sketch.sketchCurves.sketchLines

    S_sp = PL.startSketchPoint
    S, E = S_sp.geometry, PL.endSketchPoint.geometry
    L = S.distanceTo(E)
    if L < 1e-6:
        raise RackError('始点と終点が同じ位置です')
    ux, uy = (E.x - S.x) / L, (E.y - S.y) / L
    side = -1.0 if prm['flip'] else 1.0
    nx, ny = -uy * side, ux * side

    def to_sk(x, y):
        return adsk.core.Point3D.create(S.x + ux * x + nx * y, S.y + uy * x + ny * y, 0.0)

    p, s, t, ha, hf = d['p'], d['s'], d['t'], d['ha'], d['hf']
    tooth_start = prm['start'] == START_TOOTH
    phase = (-p / 2 if tooth_start else 0.0) + prm['phase']
    N = int(prm.get('teeth') or 0)
    if N <= 0:
        N = max(1, int(math.floor((L - phase) / p + 1e-9)))
    y_p = {REF_PITCH: 0.0, REF_ROOT: hf, REF_TIP: -ha}[prm['ref']]
    root_on_pl = prm['ref'] == REF_ROOT
    tip_on_pl = prm['ref'] == REF_TIP
    if prm['closed'] and prm['back'] <= hf + 1e-6:
        raise RackError('背面の位置は歯底より下(ピッチ線から歯元のたけより大きい距離)にしてください')

    # --- 歯1ピッチ分 (歯溝中心→歯溝中心) の形を先に計算 (ここまでで失敗すれば何も変更しない) ---
    xc = phase + p / 2
    verts = [(phase, -hf, 0.0),
             (xc - s / 2 - hf * t, -hf, d['rr']),
             (xc - s / 2 + ha * t, ha, 0.0),
             (xc + s / 2 - ha * t, ha, 0.0),
             (xc + s / 2 + hf * t, -hf, d['rr']),
             (phase + p, -hf, 0.0)]
    prims = shift_y(fillet_path(verts), y_p)
    if dry_run:
        return {'teeth': N, 'pitch': p, 'length': L}

    pre = prefix or _next_prefix(des)
    nn = pre + '_N'
    des.userParameters.add(nn, VI(str(N)), '', 'ラック {} の歯数'.format(pre))

    created = []
    lock = prm.get('lock', True)
    start_at_S = lock and root_on_pl and abs(phase) < 1e-9
    ch = _Chain(sketch, to_sk, S_sp if start_at_S else to_sk(*prim_start(prims[0])))
    for q in prims:
        ch.prim(q)
    seed = list(ch.curves)
    created.extend(seed)
    G0, G1 = ch.first, ch.cur
    seg = [c for c in seed if c.objectType == adsk.fusion.SketchLine.classType()]
    arcs = [c for c in seed if c.objectType == adsk.fusion.SketchArc.classType()]
    L1, L2, L3, L4, L5 = seg

    # --- 拘束 ---
    if root_on_pl and not start_at_S:
        gc.addCollinear(L1, PL)
    else:
        gc.addParallel(L1, PL)
    gc.addCollinear(L5, L1)
    if tip_on_pl:
        gc.addCollinear(L3, PL)
    else:
        gc.addParallel(L3, PL)
    if arcs:
        F1, F2 = arcs
        gc.addTangent(F1, L1)
        gc.addTangent(F1, L2)
        gc.addTangent(F2, L4)
        gc.addTangent(F2, L5)
        gc.addEqual(F2, F1)
    gc.addEqual(L5, L1)

    # --- 寸法 ---
    if not root_on_pl:
        dims.addOffsetDimension(PL, L1, to_sk(phase - 0.3 * p, (y_p - hf) / 2)).parameter.expression = _mm(abs(y_p - hf))
    if not tip_on_pl:
        dims.addOffsetDimension(PL, L3, to_sk(xc, y_p + ha + 0.5 * p)).parameter.expression = _mm(abs(y_p + ha))
    for ln, x in ((L2, xc - s / 2), (L4, xc + s / 2)):
        ad = dims.addAngularDimension(PL, ln, to_sk(x, y_p + 0.3 * ha))
        v = ad.parameter.value
        minus = abs(v - (math.pi / 2 - prm['pa'])) < abs(v - (math.pi / 2 + prm['pa']))
        ad.parameter.expression = _deg(math.pi / 2 - prm['pa'] if minus else math.pi / 2 + prm['pa'])
    if arcs:
        dims.addRadialDimension(arcs[0], to_sk(xc - s / 2 - hf * t, y_p - hf - 0.3 * p)).parameter.expression = _mm(d['rr'])
    dims.addDistanceDimension(L3.startSketchPoint, L3.endSketchPoint, AL,
                              to_sk(xc, y_p + ha + 0.25 * p)).parameter.expression = _mm(d['tip_w'])
    dims.addDistanceDimension(G0, G1, AL, to_sk(xc, y_p - hf - 0.6 * p)).parameter.expression = _mm(p)

    # --- ピッチ線上の位置 (位相) ---
    if lock and not start_at_S:
        if abs(phase) < 1e-9:
            fl = lines.addByTwoPoints(S_sp, G0)
            fl.isConstruction = True
            gc.addPerpendicular(fl, PL)
            created.append(fl)
        elif root_on_pl:
            dims.addDistanceDimension(S_sp, G0, AL, to_sk(phase / 2, -0.8 * p)).parameter.expression = _mm(abs(phase))
        else:
            fl = lines.addByTwoPoints(G0, to_sk(phase, 0.0))
            fl.isConstruction = True
            gc.addPerpendicular(fl, PL)
            gc.addCoincident(fl.endSketchPoint, PL)
            dims.addDistanceDimension(S_sp, fl.endSketchPoint, AL,
                                      to_sk(phase / 2, 0.8 * p)).parameter.expression = _mm(abs(phase))
            created.append(fl)

    # --- 背面の輪郭 ---
    if prm['closed']:
        yb = y_p - prm['back']
        el = lines.addByTwoPoints(G0, to_sk(phase, yb))
        bl = lines.addByTwoPoints(el.endSketchPoint, to_sk(phase + N * p, yb))
        er = lines.addByTwoPoints(bl.endSketchPoint, to_sk(phase + N * p, y_p - hf))
        gc.addPerpendicular(el, PL)
        gc.addParallel(bl, PL)
        gc.addPerpendicular(er, PL)
        gc.addEqual(er, el)
        dims.addOffsetDimension(PL, bl, to_sk(phase - 0.3 * p, (y_p + yb) / 2)).parameter.expression = _mm(prm['back'])
        dims.addDistanceDimension(bl.startSketchPoint, bl.endSketchPoint, AL,
                                  to_sk(phase + N * p / 2, yb - 0.5 * p)).parameter.expression = \
            '{} * {}'.format(nn, _mm(p))
        created.extend([el, bl, er])

    # --- ピッチ線方向の関連付きパターン ---
    pin = gc.createRectangularPatternInput(seed, adsk.fusion.PatternDistanceType.SpacingPatternDistanceType)
    pin.setDirectionOne(PL, VI(nn), VI(_mm(p)))
    gc.addRectangularPattern(pin)

    for c in created:
        _tag(c, pre)
    keep = {k: prm[k] for k in ('module', 'pa', 'ha', 'hf', 'root_r', 'thin', 'ref', 'start', 'phase',
                                'flip', 'closed', 'back', 'lock')}
    keep['prefix'] = pre
    a = PL.attributes.itemByName(ATTR_GROUP, ATTR_RACK)
    if a:
        a.value = json.dumps(keep)
    else:
        PL.attributes.add(ATTR_GROUP, ATTR_RACK, json.dumps(keep))
    return {'prefix': pre, 'teeth': N, 'pitch': p, 'length': L}
# ==== GEOMETRY END ====


# ---------------------------------------------------------------------------
# コマンド UI
# ---------------------------------------------------------------------------

CMD_ID = 'GearSketch_Rack'
CMD_NAME = 'ラック (モジュール指定)'
PANEL_ID = 'SketchCreatePanel'

_app = None
_ui = None
_handlers = []


def _active_sketch():
    obj = _app.activeEditObject
    return adsk.fusion.Sketch.cast(obj) if obj else None


def _resolve_points(sel, sketch):
    """選択から (S, E, s_sp, e_sp, line) を返す。S/E はスケッチ空間の Point3D。"""
    ents = [sel.selection(i).entity for i in range(sel.selectionCount)]
    line = None
    if len(ents) == 1 and ents[0].objectType == adsk.fusion.SketchLine.classType():
        line = ents[0]
        pts = [line.startSketchPoint, line.endSketchPoint]
    elif len(ents) == 2 and all(e.objectType == adsk.fusion.SketchPoint.classType() for e in ents):
        pts = ents
    else:
        return None
    out, sps = [], []
    for sp in pts:
        if sp.parentSketch == sketch:
            out.append(sp.geometry)
            sps.append(sp)
        else:
            out.append(sketch.modelToSketchSpace(sp.worldGeometry))
            sps.append(None)
    return out[0], out[1], sps[0], sps[1], (line if line and line.parentSketch == sketch else None)


def _edit_target(sel):
    if sel.selectionCount != 1:
        return None
    try:
        return find_rack(sel.selection(0).entity)
    except Exception:
        return None


def _load_into_inputs(inputs, target):
    """既存ラックの設定をダイアログに読み込む。"""
    sk, PL, pre, prm = target
    des = adsk.fusion.Design.cast(sk.parentComponent.parentDesign)
    for k in ('module', 'pa', 'thin', 'phase', 'back'):
        if k in prm:
            inputs.itemById(k).value = prm[k]
    for k in ('ha', 'hf', 'root_r'):
        if k in prm:
            inputs.itemById(k).value = prm[k]
    for k in ('flip', 'closed', 'lock'):
        if k in prm:
            inputs.itemById(k).value = bool(prm[k])
    for k in ('ref', 'start'):
        if k in prm:
            for it in inputs.itemById(k).listItems:
                it.isSelected = (it.name == prm[k])
    for it in inputs.itemById('mode').listItems:
        it.isSelected = (it.name == MODE_PARAM)
    up = des.userParameters.itemByName(pre + '_N')
    inputs.itemById('teeth').value = int(round(up.value)) if up else 0
    inputs.itemById('status').text = '既存のラック {} を編集します。OK で同じピッチ線上に作り直します。'.format(pre)


def _params(inp):
    v = lambda i: inp.itemById(i).value
    return {
        'module': v('module'),
        'pa': v('pa'),
        'ha': v('ha'),
        'hf': v('hf'),
        'root_r': v('root_r'),
        'tip_r': v('tip_r'),
        'thin': v('thin'),
        'ref': inp.itemById('ref').selectedItem.name,
        'start': inp.itemById('start').selectedItem.name,
        'phase': v('phase'),
        'flip': v('flip'),
        'closed': v('closed'),
        'back': v('back'),
        'fix': v('fix'),
        'lock': v('lock'),
        'teeth': v('teeth'),
    }


def _do(inputs):
    status = inputs.itemById('status')
    sel = inputs.itemById('sel')
    sketch = _active_sketch()
    if sketch is None and sel.selectionCount > 0:
        sketch = sel.selection(0).entity.parentSketch
    if sketch is None:
        status.text = 'スケッチを編集中に実行してください'
        return False
    prm = _params(inputs)
    target = _edit_target(sel)
    if target is not None:
        sk, PL, pre, _old = target
        try:
            draw_rack_param(sk, PL, prm, prefix=pre, dry_run=True)   # 失敗するなら既存ラックを消さない
            delete_rack(sk, pre)
            info = draw_rack_param(sk, PL, prm, prefix=pre)
        except RackError as e:
            status.text = str(e)
            return False
        status.text = '{p} を作り直しました (歯 {n} 枚, ピッチ {pt:.4f} mm)'.format(
            p=pre, n=info['teeth'], pt=info['pitch'] * 10)
        return True
    r = _resolve_points(sel, sketch)
    if r is None:
        status.text = '線を1本、または点を2つ選んでください'
        return False
    S, E, s_sp, e_sp, line = r
    if inputs.itemById('mode').selectedItem.name == MODE_PARAM:
        try:
            if line is None:
                line = sketch.sketchCurves.sketchLines.addByTwoPoints(s_sp if s_sp else S, e_sp if e_sp else E)
            line.isConstruction = True
            info = draw_rack_param(sketch, line, prm)
        except RackError as e:
            status.text = str(e)
            return False
        status.text = ('{p}: 歯 {n} 枚 / ピッチ {pt:.4f} mm。歯数はパラメータ {p}_N で変更できます。'
                       'モジュール等を変えるときは、ラックを選んでこのコマンドを再実行してください。').format(
            p=info['prefix'], n=info['teeth'], pt=info['pitch'] * 10)
        return True
    try:
        curves, info = draw_rack(sketch, S, E, prm, s_sp, e_sp)
    except RackError as e:
        status.text = str(e)
        return False
    if line is not None and inputs.itemById('construct').value:
        line.isConstruction = True
    status.text = 'ピッチ {:.4f} mm / 長さ {:.3f} mm (約 {:.2f} ピッチ) / 歯 {} 枚'.format(
        info['pitch'] * 10, info['length'] * 10, info['length'] / info['pitch'], info['teeth'])
    return True


class CreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inp = cmd.commandInputs
            VI = adsk.core.ValueInput.createByString

            s = inp.addSelectionInput('sel', '始点・終点 / 既存ラック',
                                      '基準線になる線を1本、始点と終点の2点、または編集する既存ラックの線を選択')
            s.addSelectionFilter('SketchPoints')
            s.addSelectionFilter('SketchCurves')
            s.setSelectionLimits(1, 2)

            dd = inp.addDropDownCommandInput('mode', '作り方', adsk.core.DropDownStyles.TextListDropDownStyle)
            dd.listItems.add(MODE_PARAM, True)
            dd.listItems.add(MODE_FIXED, False)
            inp.addValueInput('module', 'モジュール', 'mm', VI('1 mm'))
            inp.addValueInput('pa', '圧力角', 'deg', VI('20 deg'))
            dd = inp.addDropDownCommandInput('ref', '選んだ線の位置', adsk.core.DropDownStyles.TextListDropDownStyle)
            for i, n in enumerate([REF_PITCH, REF_ROOT, REF_TIP]):
                dd.listItems.add(n, i == 0)
            inp.addBoolValueInput('flip', '歯の向きを反転', True, '', False)
            dd = inp.addDropDownCommandInput('start', '始点に合わせる位置', adsk.core.DropDownStyles.TextListDropDownStyle)
            for i, n in enumerate([START_GAP, START_TOOTH]):
                dd.listItems.add(n, i == 0)
            inp.addValueInput('phase', '位相オフセット', 'mm', VI('0 mm'))
            inp.addBoolValueInput('lock', '歯の位置をピッチ線の始点に連動', True, '', True)
            inp.addIntegerSpinnerCommandInput('teeth', '歯数 (0=線の長さから自動)', 0, 100000, 1, 0)
            inp.addBoolValueInput('closed', '背面まで閉じた輪郭にする', True, '', True)
            inp.addValueInput('back', '背面の位置 (ピッチ線から)', 'mm', VI('5 mm'))

            g = inp.addGroupCommandInput('adv', '詳細 (係数はモジュール倍)')
            g.isExpanded = False
            gi = g.children
            gi.addFloatSpinnerCommandInput('ha', '歯末のたけ係数', '', 0.0, 5.0, 0.05, 1.0)
            gi.addFloatSpinnerCommandInput('hf', '歯元のたけ係数', '', 0.1, 5.0, 0.05, 1.25)
            gi.addFloatSpinnerCommandInput('root_r', '歯元R係数', '', 0.0, 1.0, 0.01, 0.38)
            gi.addFloatSpinnerCommandInput('tip_r', '歯先R係数 (固定形状のみ)', '', 0.0, 1.0, 0.01, 0.0)
            gi.addValueInput('thin', '歯厚減少量 (バックラッシ用)', 'mm', VI('0 mm'))
            gi.addBoolValueInput('fix', '作成した線をその場に固定 (固定形状のみ)', True, '', False)
            gi.addBoolValueInput('construct', '選んだ線を補助線にする', True, '', True)

            inp.addTextBoxCommandInput('status', '情報', '', 2, True)

            for h, ev in ((ExecuteHandler(), cmd.execute),
                          (PreviewHandler(), cmd.executePreview),
                          (ValidateHandler(), cmd.validateInputs),
                          (InputChangedHandler(), cmd.inputChanged),
                          (ActivateHandler(), cmd.activate)):
                ev.add(h)
                _handlers.append(h)
        except Exception:
            _ui.messageBox('UI作成に失敗:\n' + traceback.format_exc())


class InputChangedHandler(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        try:
            if args.input.id != 'sel':
                return
            target = _edit_target(args.inputs.itemById('sel'))
            if target is not None:
                _load_into_inputs(args.inputs, target)
        except Exception:
            pass


class ActivateHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            inputs = args.command.commandInputs
            target = _edit_target(inputs.itemById('sel'))
            if target is not None:
                _load_into_inputs(inputs, target)
        except Exception:
            pass


class ExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            inputs = args.command.commandInputs
            if not _do(inputs):
                _ui.messageBox(inputs.itemById('status').text)
        except Exception:
            _ui.messageBox('ラックの作成に失敗しました:\n' + traceback.format_exc())


class PreviewHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        inputs = args.command.commandInputs
        try:
            args.isValidResult = _do(inputs)
        except Exception as e:
            inputs.itemById('status').text = 'エラー: {}'.format(e)


class ValidateHandler(adsk.core.ValidateInputsEventHandler):
    def notify(self, args):
        try:
            i = args.inputs
            n = i.itemById('sel').selectionCount
            args.areInputsValid = n in (1, 2) and i.itemById('module').value > 0
        except Exception:
            args.areInputsValid = False


def run(context):
    global _app, _ui
    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface
        d = _ui.commandDefinitions.itemById(CMD_ID)
        if d:
            d.deleteMe()
        d = _ui.commandDefinitions.addButtonDefinition(
            CMD_ID, CMD_NAME, '選んだ2点の間に、指定モジュールのラック歯形をスケッチします')
        h = CreatedHandler()
        d.commandCreated.add(h)
        _handlers.append(h)
        panel = _ui.allToolbarPanels.itemById(PANEL_ID)
        if panel and not panel.controls.itemById(CMD_ID):
            panel.controls.addCommand(d)
    except Exception:
        if _ui:
            _ui.messageBox('GearSketch の起動に失敗:\n' + traceback.format_exc())


def stop(context):
    try:
        panel = _ui.allToolbarPanels.itemById(PANEL_ID)
        c = panel.controls.itemById(CMD_ID) if panel else None
        if c:
            c.deleteMe()
        d = _ui.commandDefinitions.itemById(CMD_ID)
        if d:
            d.deleteMe()
    except Exception:
        pass
