
import matplotlib as mpl


def streamLinePlot(fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options):
    # Compute streamlines from the grid quantity, which is assumed to be a vector field with shape (numGridPoints, 2)
    X = gridState.positions[:,0].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T
    Y = gridState.positions[:,1].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T
    U = gridQuantity[:,0].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T
    V = gridQuantity[:,1].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T

    # print("X: ", X)
    # print("Y: ", Y)
    # print("U: ", U)
    # print("V: ", V)


    strm = axis.streamplot(X, Y, U, V, color = 'k', density=1.5, linewidth=0.5)

    return strm

def updateStreamLinePlot(streamLines, fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options):# 
    for patch in axis.patches:
        if isinstance(patch, mpl.patches.FancyArrowPatch):
            # print("Removing patch: ", patch)
            patch.remove()
    # print("Patch: ", patch)

    streamLines.lines.remove()
    keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
    # Patches doesnt have a setter so we cant do this anymore
    # axis.patches = [patch for patch in axis.patches if keep(patch)]

    # cant do this as streamlines.lines is not iterable
    # Remove the old streamlines
    # for line in streamLines.lines:
        # line.remove()

    # print("Removing old streamlines")


    # Compute streamlines from the grid quantity, which is assumed to be a vector field with shape (numGridPoints, 2)
    X = gridState.positions[:,0].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T
    Y = gridState.positions[:,1].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T
    U = gridQuantity[:,0].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T
    V = gridQuantity[:,1].reshape(gridResolution[0], gridResolution[1]).cpu().numpy().T

    # print("X: ", X)
    # print("Y: ", Y)
    # print("U: ", U)
    # print("V: ", V)


    strm = axis.streamplot(X, Y, U, V, color = 'k', density=1.5, linewidth=0.5)

    return strm