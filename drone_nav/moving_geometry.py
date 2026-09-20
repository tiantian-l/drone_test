"""Conservative swept-volume checks for kinematic cylinders (no dependencies)."""


def segment_hits_box(start, end, box):
    """Closed segment vs closed axis-aligned rectangle using slab intervals."""
    enter, leave = 0.0, 1.0
    for axis, (low, high) in enumerate(((box[0], box[1]), (box[2], box[3]))):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not low <= start[axis] <= high:
                return False
        else:
            a, b = (low - start[axis]) / delta, (high - start[axis]) / delta
            enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
            if enter > leave:
                return False
    return True


def advance_cylinders(specs, boxes, half_extents, gap, dt):
    """Return new [x,y,r,h,vx,vy] states without obstacle tunnelling.

    Static boxes are inflated by radius + gap (conservative at corners).
    Dynamic peers reserve their whole swept AABB, including their old position.
    Proposals are processed in stable order; rejected moves stay and reverse.
    This is kinematic backtracking, not a physically elastic collision model.
    """
    result = [list(s) for s in specs]
    reservations = [(s[0], s[0], s[1], s[1], s[2]) for s in specs]
    mx, my = half_extents
    for i, s in enumerate(specs):
        x, y, radius, height, vx, vy = s
        end = (x + vx * dt, y + vy * dt)
        blocked = not (-mx + radius <= end[0] <= mx - radius and
                       -my + radius <= end[1] <= my - radius)
        for ox, oy, wx, wy, _ in boxes:
            pad = radius + gap
            rect = (ox - wx / 2 - pad, ox + wx / 2 + pad,
                    oy - wy / 2 - pad, oy + wy / 2 + pad)
            if segment_hits_box((x, y), end, rect):
                blocked = True
                break
        if not blocked:
            for j, (xl, xh, yl, yh, other_radius) in enumerate(reservations):
                if i == j:
                    continue
                pad = radius + other_radius + gap
                if segment_hits_box((x, y), end,
                                    (xl - pad, xh + pad, yl - pad, yh + pad)):
                    blocked = True
                    break
        if blocked:
            result[i][4:6] = [-vx, -vy]
        else:
            result[i][0:2] = end
            reservations[i] = (min(x, end[0]), max(x, end[0]),
                               min(y, end[1]), max(y, end[1]), radius)
    return result
