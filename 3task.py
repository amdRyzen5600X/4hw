def relaxation(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, tolerance=1e-6, max_iterations=1000):
    x = y = z = 0

    for _ in range(max_iterations):

        Rx = d1 - a1 * x - b1 * y - c1 * z
        Ry = d2 - a2 * x - b2 * y - c2 * z
        Rz = d3 - a3 * x - b3 * y - c3 * z

        if abs(Rx) < tolerance and abs(Ry) < tolerance and abs(Rz) < tolerance:
            break

        max_residual = max(abs(Rx), abs(Ry), abs(Rz))
        if max_residual == abs(Rx):
            x += Rx / a1
        elif max_residual == abs(Ry):
            y += Ry / b2
        else:
            z += Rz / c3
    return x, y, z
