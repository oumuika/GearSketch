# -*- coding: utf-8 -*-
"""
GearSketch - Autodesk Fusion アドイン
スケッチ編集中に「作成」メニューから、任意モジュールのギアをスケッチします。

  ラック (モジュール指定)
    - 選んだ線(または2点)を基準線にしてラック歯形を描く
    - パラメトリック: 歯1枚を拘束で定義してピッチ線方向にパターン → ピッチ線と一体で動く
    - 固定形状: 選んだ2点の位置で歯形を正確に切る

  平歯車 (モジュール指定)
    - 選んだ点(または円の中心)を中心に、インボリュート歯形の平歯車を描く
    - 歯1枚を中心点と「歯の中心線」に対して拘束し、円形パターンで並べる
      → 中心を動かすと移動、中心線を回すと回転、歯をドラッグすると一体で動く
    - ピッチ円(補助円)を作るので、ラックのピッチ線や相手ギアのピッチ円と拘束して噛み合わせられる

  どちらも、既存のラック/ギアの線を選んでコマンドを実行すると設定を読み込み、作り直せます。

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


# ===========================================================================
# 平歯車
#   ・歯1枚 (歯溝中心→歯溝中心) を、中心点と「歯の中心線」(補助線) に対して寸法と対称拘束で固定し、
#     中心点まわりの円形パターンで並べる
#   ・中心点を動かすと移動、中心線を回すと回転、歯をドラッグすると全体が一体で動く
#   ・ピッチ円 (補助円) を作るので、ラックのピッチ線や相手ギアのピッチ円と拘束して噛み合わせられる
#   ・歯数やモジュールの変更は、ギアを選んでコマンドを再実行 → 中心・中心線・ピッチ円を残して作り直す
# ===========================================================================

ATTR_GEAR = 'gear'   # 歯の中心線に付ける (値 = JSON パラメータ)


class GearError(RackError):
    pass


def _inv(a):
    return math.tan(a) - a


def spur_dims(prm):
    m, z = prm['module'], int(prm['teeth'])
    a = prm['pa']
    x = prm.get('shift', 0.0)
    d = {'m': m, 'z': z, 'a': a, 'x': x}
    d['r'] = m * z / 2.0
    d['rb'] = d['r'] * math.cos(a)
    d['ra'] = d['r'] + m * (prm.get('ha', 1.0) + x)
    d['rf'] = d['r'] - m * (prm.get('hf', 1.25) - x)
    d['rho'] = prm.get('root_r', 0.38) * m
    s = m * (math.pi / 2 + 2 * x * math.tan(a)) - prm.get('thin', 0.0)   # ピッチ円上の歯厚
    d['psi'] = s / (2 * d['r'])
    d['gap'] = math.pi / z
    return d


def _flank_phi(d, R):
    """下半分の歯面上、半径 R の点の角度。R>=rb はインボリュート、それ未満は基礎円から半径方向の直線。"""
    rb = d['rb']
    if R <= rb:
        return -(d['psi'] + _inv(d['a']))
    return -(d['psi'] + _inv(d['a']) - _inv(math.acos(rb / R)))


def _pol(R, phi):
    return (R * math.cos(phi), R * math.sin(phi))


def spur_half(prm, n_fit=7):
    """歯の中心線を +X とした局所座標で、歯の下半分 (φ<0) の要所を返す。"""
    if prm['module'] <= 0:
        raise GearError('モジュールは正の値にしてください')
    if int(prm['teeth']) < 4:
        raise GearError('歯数は 4 以上にしてください')
    if not (math.radians(0.5) < prm['pa'] < math.radians(45)):
        raise GearError('圧力角は 0.5°〜45° の範囲にしてください')
    d = spur_dims(prm)
    rb, ra, rf, rho, gap = d['rb'], d['ra'], d['rf'], d['rho'], d['gap']
    if rf <= 0:
        raise GearError('歯底円の半径が 0 以下になります (歯数が少なすぎるか、歯元のたけが大きすぎます)')
    if _flank_phi(d, ra) >= -1e-9:
        raise GearError('歯先が尖ってしまいます (転位係数か歯末のたけを小さくしてください)')
    phi_foot = _flank_phi(d, rf)
    if phi_foot <= -gap + 1e-9:
        raise GearError('歯溝がなくなります (歯数・圧力角・転位係数・歯厚減少量を見直してください)')

    out = {'d': d, 'G': _pol(rf, -gap)}
    if rho > 1e-12:
        # 歯面を細かい折れ線にして、歯底円と歯面の両方に接する半径 rho の円を二分法で求める
        pts = [(rf, _flank_phi(d, rf))] if rf < rb else []
        Rs = max(rf, rb)
        ts, ta = math.sqrt(max(0.0, (Rs / rb) ** 2 - 1)), math.sqrt((ra / rb) ** 2 - 1)
        for i in range(201):
            t = ts + (ta - ts) * i / 200
            R = rb * math.sqrt(1 + t * t)
            pts.append((R, _flank_phi(d, R)))
        poly = [_pol(R, p) for R, p in pts]

        def nearest(c):
            best, bq = 1e9, None
            for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
                dx, dy = x2 - x1, y2 - y1
                L2 = dx * dx + dy * dy
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((c[0] - x1) * dx + (c[1] - y1) * dy) / L2))
                q = (x1 + t * dx, y1 + t * dy)
                dd = math.hypot(q[0] - c[0], q[1] - c[1])
                if dd < best:
                    best, bq = dd, q
            return best, bq

        Rc = rf + rho
        lo, hi = -gap, _flank_phi(d, Rc)
        if nearest(_pol(Rc, lo))[0] < rho:
            raise GearError('歯元R が大きすぎて歯溝に収まりません。歯元R係数を小さくしてください。')
        for _ in range(80):
            mid = (lo + hi) / 2
            if nearest(_pol(Rc, mid))[0] > rho:
                lo = mid
            else:
                hi = mid
        phic = (lo + hi) / 2
        Fc = _pol(Rc, phic)
        RF = math.hypot(*nearest(Fc)[1])
        out.update(Rt=_pol(rf, phic), Fc=Fc, Ft=_pol(RF, _flank_phi(d, RF)), R_foot=RF)
    else:
        out.update(Rt=None, Fc=None, Ft=_pol(rf, phi_foot), R_foot=rf)

    if out['R_foot'] < rb - 1e-9:
        out['B'] = _pol(rb, _flank_phi(d, rb))
        Rs = rb
    else:
        out['B'] = None
        Rs = out['R_foot']
    ts, ta = math.sqrt(max(0.0, (Rs / rb) ** 2 - 1)), math.sqrt((ra / rb) ** 2 - 1)
    fit = []
    for i in range(n_fit):
        t = ts + (ta - ts) * i / (n_fit - 1)
        R = rb * math.sqrt(1 + t * t) if i < n_fit - 1 else ra
        fit.append(_pol(R, _flank_phi(d, R)))
    out['fit'] = fit            # fit[0] = 歯面の始点 (B か Ft), fit[-1] = 歯先 T
    out['T'] = fit[-1]
    return out


def _fillet_mid(Fc, Rt, Ft):
    a0 = math.atan2(Rt[1] - Fc[1], Rt[0] - Fc[0])
    a1 = math.atan2(Ft[1] - Fc[1], Ft[0] - Fc[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    am = a0 + da / 2
    r = math.hypot(Rt[0] - Fc[0], Rt[1] - Fc[1])
    return (Fc[0] + r * math.cos(am), Fc[1] + r * math.sin(am))


def _nearest_end(curves, p3):
    best, bp = 1e9, None
    for c in curves:
        for sp in (c.startSketchPoint, c.endSketchPoint):
            dd = sp.geometry.distanceTo(p3)
            if dd < best:
                best, bp = dd, sp
    return bp


def _nearest_center(curves, p3):
    best, bp = 1e9, None
    for c in curves:
        if c.objectType == adsk.fusion.SketchArc.classType():
            dd = c.centerSketchPoint.geometry.distanceTo(p3)
            if dd < best:
                best, bp = dd, c.centerSketchPoint
    return bp


def _next_gear_prefix(des):
    used = set()
    for a in des.findAttributes(ATTR_GROUP, ATTR_GEAR):
        try:
            used.add(json.loads(a.value).get('prefix'))
        except Exception:
            pass
    i = 1
    while 'Gear{}'.format(i) in used:
        i += 1
    return 'Gear{}'.format(i)


def draw_spur_param(sketch, C_sp, prm, prefix=None, centerline=None, pitch=None, dry_run=False):
    """C_sp: 中心の SketchPoint。centerline / pitch: 作り直しのときに引き継ぐ中心線とピッチ円。"""
    h = spur_half(prm)
    d = h['d']
    if dry_run:
        return {'teeth': d['z']}
    des = adsk.fusion.Design.cast(sketch.parentComponent.parentDesign)
    pre = prefix or _next_gear_prefix(des)
    VI = adsk.core.ValueInput.createByString
    AL = adsk.fusion.DimensionOrientations.AlignedDimensionOrientation
    gc = sketch.geometricConstraints
    dims = sketch.sketchDimensions
    lines = sketch.sketchCurves.sketchLines
    arcs = sketch.sketchCurves.sketchArcs
    circles = sketch.sketchCurves.sketchCircles
    C = C_sp.geometry
    created = []

    # --- ピッチ円 (補助) と歯の中心線 (補助, ギアの回転を表す) ---
    if pitch is None:
        pitch = circles.addByCenterRadius(C_sp, d['r'])
        pitch.isConstruction = True
        dims.addDiameterDimension(pitch, adsk.core.Point3D.create(C.x + d['r'] * 0.7, C.y - d['r'] * 0.7, 0))
    elif not _set_pitch_diameter(sketch, pitch, d['r']):
        dims.addDiameterDimension(pitch, adsk.core.Point3D.create(C.x + d['r'] * 0.7, C.y - d['r'] * 0.7, 0))
    if centerline is None:
        th = prm.get('angle', math.pi / 2)
        e = adsk.core.Point3D.create(C.x + d['r'] * math.cos(th), C.y + d['r'] * math.sin(th), 0)
        centerline = lines.addByTwoPoints(C_sp, e)
        centerline.isConstruction = True
        gc.addCoincident(centerline.endSketchPoint, pitch)
    s0, s1 = centerline.startSketchPoint.geometry, centerline.endSketchPoint.geometry
    th = math.atan2(s1.y - s0.y, s1.x - s0.x)
    ct, st = math.cos(th), math.sin(th)

    def P(q, mirror=False):
        x, y = q[0], (-q[1] if mirror else q[1])
        return adsk.core.Point3D.create(C.x + x * ct - y * st, C.y + x * st + y * ct, 0)

    def mid_arc(R, a0, a1):
        return _pol(R, (a0 + a1) / 2)

    def ang(q):
        return math.atan2(q[1], q[0])

    def tp(q, mirror=False, k=1.25):
        return P((q[0] * k, q[1] * k), mirror)

    st_ = {'cur': None}

    def arc_to(mid, end, mirror=False):
        endp = P(end, mirror)
        a = arcs.addByThreePoints(st_['cur'], P(mid, mirror), endp)
        s, e = a.startSketchPoint, a.endSketchPoint
        st_['cur'] = s if s.geometry.distanceTo(endp) < e.geometry.distanceTo(endp) else e
        created.append(a)
        return a

    def line_to(end, mirror=False):
        ln = lines.addByTwoPoints(st_['cur'], P(end, mirror))
        st_['cur'] = ln.endSketchPoint
        created.append(ln)
        return ln

    def spline_to(pts, mirror=False):
        oc = adsk.core.ObjectCollection.create()
        oc.add(st_['cur'])
        for q in pts[1:]:
            oc.add(P(q, mirror))
        sp = sketch.sketchCurves.sketchFittedSplines.add(oc)
        created.append(sp)
        fp = sp.fitPoints
        st_['cur'] = fp.item(fp.count - 1)
        return [fp.item(i) for i in range(fp.count)]

    def dist_dim(pt, q, mirror=False):
        dims.addDistanceDimension(C_sp, pt, AL, tp(q, mirror, 0.6))

    def off_dim(pt, q, mirror=False):
        dims.addOffsetDimension(centerline, pt, tp(q, mirror, 1.1))

    rf, ra = d['rf'], d['ra']
    G, Rt, Fc, Ft, B, fit, T = h['G'], h['Rt'], h['Fc'], h['Ft'], h['B'], h['fit'], h['T']

    # ===== 下半分: 歯底円弧 → 歯元R → (基礎円までの直線) → インボリュート =====
    st_['cur'] = P(G)
    foot = Rt if Fc else Ft
    a_root = arc_to(mid_arc(rf, ang(G), ang(foot)), foot)
    gpt = a_root.startSketchPoint if a_root.endSketchPoint == st_['cur'] else a_root.endSketchPoint
    gc.addCoincident(a_root.centerSketchPoint, C_sp)
    dist_dim(gpt, G)
    off_dim(gpt, G)
    off_dim(st_['cur'], foot)
    if Fc:
        a_f = arc_to(_fillet_mid(Fc, Rt, Ft), Ft)
        dist_dim(a_f.centerSketchPoint, Fc)
        off_dim(a_f.centerSketchPoint, Fc)
        off_dim(st_['cur'], Ft)
    if B:
        line_to(B)
        dist_dim(st_['cur'], B)
        off_dim(st_['cur'], B)
    low_pts = spline_to(fit)
    for pt, q in zip(low_pts[1:], fit[1:]):
        dist_dim(pt, q)
        off_dim(pt, q)

    # ===== 歯先円弧 =====
    a_tip = arc_to((ra, 0.0), T, mirror=True)
    gc.addCoincident(a_tip.centerSketchPoint, C_sp)
    off_dim(st_['cur'], T, mirror=True)

    # ===== 上半分: 中心線に対して対称 =====
    up_pts = list(reversed(spline_to(list(reversed(fit)), mirror=True)))
    for i in range(1, len(fit) - 1):
        gc.addSymmetry(low_pts[i], up_pts[i], centerline)
    if B:
        gc.addSymmetry(low_pts[0], up_pts[0], centerline)
        line_to(Ft, mirror=True)
    gc.addSymmetry(_nearest_end(created, P(Ft)), st_['cur'], centerline)
    if Fc:
        a_fu = arc_to(_fillet_mid(Fc, Rt, Ft), Rt, mirror=True)
        gc.addSymmetry(_nearest_center(created[:-1], P(Fc)), a_fu.centerSketchPoint, centerline)
        off_dim(st_['cur'], Rt, mirror=True)
    a_root_u = arc_to(mid_arc(rf, ang(foot), ang(G)), G, mirror=True)
    gc.addCoincident(a_root_u.centerSketchPoint, C_sp)
    off_dim(st_['cur'], G, mirror=True)
    seed = list(created)

    # ===== 中心まわりの円形パターン =====
    pin = gc.createCircularPatternInput(seed, C_sp)
    pin.quantity = VI(str(d['z']))
    pin.totalAngle = VI('360 deg')
    gc.addCircularPattern(pin)

    # ===== 軸穴 =====
    if prm.get('hole', 0) > 1e-9:
        hc = circles.addByCenterRadius(C_sp, prm['hole'] / 2)
        dims.addDiameterDimension(hc, adsk.core.Point3D.create(C.x + prm['hole'] * 0.3, C.y + prm['hole'] * 0.3, 0))
        created.append(hc)

    for c in created:
        _tag(c, pre)
    keep = {k: prm[k] for k in ('module', 'teeth', 'pa', 'shift', 'ha', 'hf', 'root_r', 'thin', 'hole', 'angle') if k in prm}
    keep.update(prefix=pre, mode='param')
    _set_gear_attr(centerline, keep, others=(pitch,))
    return {'prefix': pre, 'teeth': d['z'], 'd': d, 'centerline': centerline, 'pitch': pitch}


def _set_gear_attr(holder, keep, others=()):
    for o in others:
        try:
            a = o.attributes.itemByName(ATTR_GROUP, ATTR_GEAR) if o else None
            if a:
                a.deleteMe()
        except Exception:
            pass
    a = holder.attributes.itemByName(ATTR_GROUP, ATTR_GEAR)
    if a:
        a.value = json.dumps(keep)
    else:
        holder.attributes.add(ATTR_GROUP, ATTR_GEAR, json.dumps(keep))


def _spur_tooth_segments(h):
    """歯1枚 (歯溝中心→次の歯溝中心) を局所座標の区間列にする。
    ('A', mid, end) 円弧 / ('L', end) 直線 / ('S', [点...]) スプライン (始点は直前の終点)"""
    d = h['d']
    rf, ra = d['rf'], d['ra']
    G, Rt, Fc, Ft, B, fit, T = h['G'], h['Rt'], h['Fc'], h['Ft'], h['B'], h['fit'], h['T']
    mir = lambda q: (q[0], -q[1])
    ang = lambda q: math.atan2(q[1], q[0])
    foot = Rt if Fc else Ft
    seg = [('A', _pol(rf, (ang(G) + ang(foot)) / 2), foot)]
    if Fc:
        seg.append(('A', _fillet_mid(Fc, Rt, Ft), Ft))
    if B:
        seg.append(('L', B))
    seg.append(('S', fit[1:]))
    seg.append(('A', (ra, 0.0), mir(T)))
    seg.append(('S', [mir(q) for q in reversed(fit[:-1])]))
    if B:
        seg.append(('L', mir(Ft)))
    if Fc:
        seg.append(('A', mir(_fillet_mid(Fc, Rt, Ft)), mir(Rt)))
    seg.append(('A', mir(_pol(rf, (ang(G) + ang(foot)) / 2)), mir(G)))
    return seg


def draw_spur_fixed(sketch, C_sp, prm, prefix=None, pitch=None, centerline=None, dry_run=False):
    """拘束を付けずに全歯を描く (固定形状)。中心とピッチ円(補助)だけ共有する。"""
    h = spur_half(prm)
    d = h['d']
    if dry_run:
        return {'teeth': d['z']}
    des = adsk.fusion.Design.cast(sketch.parentComponent.parentDesign)
    pre = prefix or _next_gear_prefix(des)
    circles = sketch.sketchCurves.sketchCircles
    C = C_sp.geometry
    if pitch is None:
        pitch = circles.addByCenterRadius(C_sp, d['r'])
        pitch.isConstruction = True
    else:
        _set_pitch_diameter(sketch, pitch, d['r'])
    if centerline is not None:
        s0, s1 = centerline.startSketchPoint.geometry, centerline.endSketchPoint.geometry
        th0 = math.atan2(s1.y - s0.y, s1.x - s0.x)
    else:
        th0 = prm.get('angle', math.pi / 2)
    seg = _spur_tooth_segments(h)
    created = []
    lines = sketch.sketchCurves.sketchLines
    arcs = sketch.sketchCurves.sketchArcs
    was = sketch.isComputeDeferred
    sketch.isComputeDeferred = True
    try:
        cur = None
        first = None
        z = d['z']
        for k in range(z):
            th = th0 + 2 * math.pi * k / z
            ct, st = math.cos(th), math.sin(th)

            def P(q):
                return adsk.core.Point3D.create(C.x + q[0] * ct - q[1] * st, C.y + q[0] * st + q[1] * ct, 0)
            if cur is None:
                cur = P(h['G'])
            for i, sg in enumerate(seg):
                last = (k == z - 1 and i == len(seg) - 1)
                if sg[0] == 'A':
                    endp = first if last else P(sg[2])
                    a = arcs.addByThreePoints(cur, P(sg[1]), endp)
                    tgt = endp.geometry if last else endp
                    sp_, ep_ = a.startSketchPoint, a.endSketchPoint
                    near_end = sp_ if sp_.geometry.distanceTo(tgt) < ep_.geometry.distanceTo(tgt) else ep_
                    if first is None:
                        first = ep_ if near_end == sp_ else sp_
                    cur = near_end
                    created.append(a)
                elif sg[0] == 'L':
                    ln = lines.addByTwoPoints(cur, P(sg[1]))
                    cur = ln.endSketchPoint
                    created.append(ln)
                else:
                    oc = adsk.core.ObjectCollection.create()
                    oc.add(cur)
                    for q in sg[1]:
                        oc.add(P(q))
                    sp = sketch.sketchCurves.sketchFittedSplines.add(oc)
                    cur = sp.fitPoints.item(sp.fitPoints.count - 1)
                    created.append(sp)
        if prm.get('hole', 0) > 1e-9:
            created.append(circles.addByCenterRadius(C_sp, prm['hole'] / 2))
    finally:
        sketch.isComputeDeferred = was
    for c in created:
        _tag(c, pre)
    keep = {k: prm[k] for k in ('module', 'teeth', 'pa', 'shift', 'ha', 'hf', 'root_r', 'thin', 'hole', 'angle') if k in prm}
    keep.update(prefix=pre, mode='fixed')
    _set_gear_attr(pitch, keep, others=(centerline,))
    return {'prefix': pre, 'teeth': d['z'], 'd': d, 'centerline': centerline, 'pitch': pitch}


def _set_pitch_diameter(sketch, pitch, r):
    dims = sketch.sketchDimensions
    for i in range(dims.count):
        dm = dims.item(i)
        if dm.objectType == adsk.fusion.SketchDiameterDimension.classType() and dm.entity == pitch:
            dm.parameter.value = 2 * r
            return True
    pitch.radius = r
    return False


def _gear_center_after_delete(C_sp, centerline, pitch):
    """軸穴の円などを消すと、共有していた中心点の参照が無効になることがあるので取り直す。"""
    for get in (lambda: pitch.centerSketchPoint, lambda: centerline.startSketchPoint, lambda: C_sp):
        try:
            p = get()
            if p is not None and p.isValid:
                return p
        except Exception:
            continue
    raise GearError('ギアの中心点が見つかりません')


def _pattern_owner_prefix(entity, sketch):
    """パターンで増えた要素なら、元の要素のプレフィックスを返す。"""
    pat_types = (adsk.fusion.RectangularPatternConstraint.classType(),
                 adsk.fusion.CircularPatternConstraint.classType())
    for i in range(sketch.geometricConstraints.count):
        g = sketch.geometricConstraints.item(i)
        if g.objectType not in pat_types:
            continue
        try:
            ents = list(g.entities)
            pre = _attr(ents[0], ATTR_MEMBER) if ents else None
            if pre and any(c == entity for c in g.createdEntities):
                return pre
        except Exception:
            continue
    return None


def find_gear(entity):
    """選択要素が既存ギアの一部なら (sketch, center, centerline, pitch, prefix, params) を返す。
    パラメトリックは歯の中心線に、固定形状はピッチ円に設定を保存している。"""
    sk = entity.parentSketch
    holder = entity if _attr(entity, ATTR_GEAR) else None
    if holder is None:
        pre = _attr(entity, ATTR_MEMBER) or _pattern_owner_prefix(entity, sk)
        if not pre:
            return None
        for coll in (sk.sketchCurves.sketchLines, sk.sketchCurves.sketchCircles):
            for c in coll:
                v = _attr(c, ATTR_GEAR)
                if v and json.loads(v).get('prefix') == pre:
                    holder = c
                    break
            if holder:
                break
        if holder is None:
            return None
    prm = json.loads(_attr(holder, ATTR_GEAR))
    if holder.objectType == adsk.fusion.SketchLine.classType():
        cl, C_sp = holder, holder.startSketchPoint
        pitch = None
        for c in sk.sketchCurves.sketchCircles:
            if c.isConstruction and c.centerSketchPoint == C_sp:
                for gcn in cl.endSketchPoint.geometricConstraints:
                    if gcn.objectType == adsk.fusion.CoincidentConstraint.classType() and gcn.entity == c:
                        pitch = c
    else:
        pitch, C_sp, cl = holder, holder.centerSketchPoint, None
        for ln in sk.sketchCurves.sketchLines:      # 以前パラメトリックだったときの中心線が残っていれば使う
            if ln.isConstruction and ln.startSketchPoint == C_sp:
                for gcn in ln.endSketchPoint.geometricConstraints:
                    if gcn.objectType == adsk.fusion.CoincidentConstraint.classType() and gcn.entity == pitch:
                        cl = ln
    return sk, C_sp, cl, pitch, prm['prefix'], prm


def delete_gear(sketch, pre):
    """中心点・中心線・ピッチ円を残して、歯・パターン・軸穴・寸法を削除する。"""
    des = adsk.fusion.Design.cast(sketch.parentComponent.parentDesign)
    members = []
    for i in range(sketch.geometricConstraints.count - 1, -1, -1):
        g = sketch.geometricConstraints.item(i)
        if g.objectType != adsk.fusion.CircularPatternConstraint.classType():
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
            dd.listItems.add(MODE_FIXED, True)
            dd.listItems.add(MODE_PARAM, False)
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


# ---------------------------------------------------------------------------
# 平歯車コマンド UI
# ---------------------------------------------------------------------------

SPUR_CMD_ID = 'GearSketch_Spur'
SPUR_CMD_NAME = '平歯車 (モジュール指定)'
SPUR_FIXED = '固定形状 (拘束なし・軽い)'
SPUR_PARAM = 'パラメトリック (まとめて動く)'


def _spur_target(sel):
    if sel.selectionCount != 1:
        return None
    try:
        return find_gear(sel.selection(0).entity)
    except Exception:
        return None


def _spur_params(inp):
    v = lambda i: inp.itemById(i).value
    return {
        'module': v('module'), 'teeth': v('teeth'), 'pa': v('pa'), 'shift': v('shift'),
        'angle': v('angle'), 'hole': v('hole'),
        'ha': v('ha'), 'hf': v('hf'), 'root_r': v('root_r'), 'thin': v('thin'),
        'mode': 'param' if inp.itemById('mode').selectedItem.name == SPUR_PARAM else 'fixed',
    }


def _spur_load(inputs, target):
    sk, C_sp, cl, pitch, pre, prm = target
    for k in ('module', 'pa', 'hole', 'thin'):
        if k in prm:
            inputs.itemById(k).value = prm[k]
    for k in ('shift', 'ha', 'hf', 'root_r'):
        if k in prm:
            inputs.itemById(k).value = prm[k]
    if 'teeth' in prm:
        inputs.itemById('teeth').value = int(prm['teeth'])
    if 'angle' in prm:
        inputs.itemById('angle').value = prm['angle']
    for it in inputs.itemById('mode').listItems:
        it.isSelected = (it.name == (SPUR_PARAM if prm.get('mode', 'param') == 'param' else SPUR_FIXED))
    inputs.itemById('status').text = '既存のギア {} を編集します。OK で同じ中心・中心線・ピッチ円のまま作り直します。'.format(pre)


def _spur_info(d, pre):
    return ('{p}: 歯数 {z} / ピッチ円直径 {dp:.3f} mm / 歯先円直径 {da:.3f} mm / 歯底円直径 {df:.3f} mm。'
            '相手ギア(歯数 z2)との中心距離は m×({z}+z2)/2 (転位なしの場合)。').format(
        p=pre, z=d['z'], dp=2 * d['r'] * 10, da=2 * d['ra'] * 10, df=2 * d['rf'] * 10)


def _spur_center(sel, sketch):
    e = sel.selection(0).entity
    t = e.objectType
    if t == adsk.fusion.SketchPoint.classType():
        if e.parentSketch == sketch:
            return e
        return sketch.sketchPoints.add(sketch.modelToSketchSpace(e.worldGeometry))
    if t in (adsk.fusion.SketchCircle.classType(), adsk.fusion.SketchArc.classType()):
        c = e.centerSketchPoint
        if e.parentSketch == sketch:
            return c
        return sketch.sketchPoints.add(sketch.modelToSketchSpace(c.worldGeometry))
    return None


def _spur_do(inputs):
    status = inputs.itemById('status')
    sel = inputs.itemById('sel')
    sketch = _active_sketch()
    if sketch is None and sel.selectionCount > 0:
        sketch = sel.selection(0).entity.parentSketch
    if sketch is None:
        status.text = 'スケッチを編集中に実行してください'
        return False
    prm = _spur_params(inputs)
    target = _spur_target(sel)
    try:
        draw = draw_spur_param if prm['mode'] == 'param' else draw_spur_fixed
        if target is not None:
            sk, C_sp, cl, pitch, pre, _old = target
            draw(sk, C_sp, prm, dry_run=True)        # 失敗するなら既存ギアを消さない
            delete_gear(sk, pre)
            C_sp = _gear_center_after_delete(C_sp, cl, pitch)
            info = draw(sk, C_sp, prm, prefix=pre, centerline=cl, pitch=pitch)
        else:
            C_sp = _spur_center(sel, sketch)
            if C_sp is None:
                status.text = '中心にする点 (または円) を選んでください'
                return False
            draw(sketch, C_sp, prm, dry_run=True)
            info = draw(sketch, C_sp, prm)
    except RackError as e:
        status.text = str(e)
        return False
    status.text = _spur_info(info['d'], info['prefix'])
    return True


class SpurCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inp = cmd.commandInputs
            VI = adsk.core.ValueInput.createByString
            s = inp.addSelectionInput('sel', '中心 / 既存ギア', '中心にする点か円、または編集する既存ギアの線を選択')
            s.addSelectionFilter('SketchPoints')
            s.addSelectionFilter('SketchCurves')
            s.setSelectionLimits(1, 1)
            dd = inp.addDropDownCommandInput('mode', '作り方', adsk.core.DropDownStyles.TextListDropDownStyle)
            dd.listItems.add(SPUR_FIXED, True)
            dd.listItems.add(SPUR_PARAM, False)
            inp.addValueInput('module', 'モジュール', 'mm', VI('1 mm'))
            inp.addIntegerSpinnerCommandInput('teeth', '歯数', 4, 10000, 1, 20)
            inp.addValueInput('pa', '圧力角', 'deg', VI('20 deg'))
            inp.addFloatSpinnerCommandInput('shift', '転位係数', '', -1.0, 2.0, 0.05, 0.0)
            inp.addValueInput('angle', '歯の向き (中心線の角度)', 'deg', VI('90 deg'))
            inp.addValueInput('hole', '軸穴の直径 (0=なし)', 'mm', VI('0 mm'))
            g = inp.addGroupCommandInput('adv', '詳細 (係数はモジュール倍)')
            g.isExpanded = False
            gi = g.children
            gi.addFloatSpinnerCommandInput('ha', '歯末のたけ係数', '', 0.0, 5.0, 0.05, 1.0)
            gi.addFloatSpinnerCommandInput('hf', '歯元のたけ係数', '', 0.1, 5.0, 0.05, 1.25)
            gi.addFloatSpinnerCommandInput('root_r', '歯元R係数', '', 0.0, 1.0, 0.01, 0.38)
            gi.addValueInput('thin', '歯厚減少量 (バックラッシ用)', 'mm', VI('0 mm'))
            inp.addTextBoxCommandInput('status', '情報', '', 3, True)
            for h, ev in ((SpurExecuteHandler(), cmd.execute),
                          (SpurPreviewHandler(), cmd.executePreview),
                          (SpurValidateHandler(), cmd.validateInputs),
                          (SpurInputChangedHandler(), cmd.inputChanged),
                          (SpurActivateHandler(), cmd.activate)):
                ev.add(h)
                _handlers.append(h)
        except Exception:
            _ui.messageBox('UI作成に失敗:\n' + traceback.format_exc())


class SpurInputChangedHandler(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        try:
            if args.input.id != 'sel':
                return
            t = _spur_target(args.inputs.itemById('sel'))
            if t is not None:
                _spur_load(args.inputs, t)
        except Exception:
            pass


class SpurActivateHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            inputs = args.command.commandInputs
            t = _spur_target(inputs.itemById('sel'))
            if t is not None:
                _spur_load(inputs, t)
        except Exception:
            pass


class SpurExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            inputs = args.command.commandInputs
            if not _spur_do(inputs):
                _ui.messageBox(inputs.itemById('status').text)
        except Exception:
            _ui.messageBox('平歯車の作成に失敗しました:\n' + traceback.format_exc())


class SpurPreviewHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        inputs = args.command.commandInputs
        try:
            args.isValidResult = _spur_do(inputs)
        except Exception as e:
            inputs.itemById('status').text = 'エラー: {}'.format(e)


class SpurValidateHandler(adsk.core.ValidateInputsEventHandler):
    def notify(self, args):
        try:
            i = args.inputs
            args.areInputsValid = i.itemById('sel').selectionCount == 1 and i.itemById('module').value > 0
        except Exception:
            args.areInputsValid = False


# ---------------------------------------------------------------------------
# 登録
# ---------------------------------------------------------------------------

import os

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
DROPDOWN_ID = 'GearSketch_DropDown'
DROPDOWN_TEXT = 'GearSketch'

_COMMANDS = [
    (SPUR_CMD_ID, SPUR_CMD_NAME, '選んだ点を中心に、指定モジュール・歯数の平歯車をスケッチします', SpurCreatedHandler, 'Spur'),
    (CMD_ID, CMD_NAME, '選んだ線(2点)に沿って、指定モジュールのラック歯形をスケッチします', CreatedHandler, 'Rack'),
]


def _res(name):
    p = os.path.join(_RES, name)
    return p if os.path.isdir(p) else ''


def run(context):
    global _app, _ui
    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface
        panel = _ui.allToolbarPanels.itemById(PANEL_ID)
        dd = None
        if panel:
            old = panel.controls.itemById(DROPDOWN_ID)
            if old:
                old.deleteMe()
            for cid, *_rest in _COMMANDS:          # 旧版でパネル直下に置いたボタンを片付ける
                c = panel.controls.itemById(cid)
                if c:
                    c.deleteMe()
            dd = panel.controls.addDropDown(DROPDOWN_TEXT, _res('GearSketch'), DROPDOWN_ID)
        for cid, name, tip, handler_cls, icon in _COMMANDS:
            d = _ui.commandDefinitions.itemById(cid)
            if d:
                d.deleteMe()
            d = _ui.commandDefinitions.addButtonDefinition(cid, name, tip, _res(icon))
            h = handler_cls()
            d.commandCreated.add(h)
            _handlers.append(h)
            if dd:
                dd.controls.addCommand(d)
    except Exception:
        if _ui:
            _ui.messageBox('GearSketch の起動に失敗:\n' + traceback.format_exc())


def stop(context):
    try:
        panel = _ui.allToolbarPanels.itemById(PANEL_ID)
        if panel:
            dd = panel.controls.itemById(DROPDOWN_ID)
            if dd:
                dd.deleteMe()
        for cid, *_rest in _COMMANDS:
            d = _ui.commandDefinitions.itemById(cid)
            if d:
                d.deleteMe()
    except Exception:
        pass
