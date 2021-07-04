import numpy as np
import mathutils
from operator import itemgetter
from lib.bpypolyskel.bpypolyskel import skeletonize
from lib.bpypolyskel.bpyeuclid import Edge2

# parameters
MAX_SLOPE    = np.tan(5./180.*np.pi)       # maximum slope (5°) of skeleton edges to be accepted as roof rodges
MAX_STRAIGHT = np.sin(5./180.*np.pi)       # maximum angle (5°) between edges to be accepted as straight angle
MARGIN       = 0.5                         # vertical safety margin of rectangle 
MINDIM       = 2.                          # minimal dimension of the rectangles 

# crossing number test for a point in a polygon
# input:  p = a point,
#         poly = ordered list of vertex points of a polygon
# return: 0 = outside, 1 = inside
def countCrossings(p, poly):
    nCrossings = 0    # the crossing number counter

    # repeat the first vertex at end
    poly = tuple(poly[:])+(poly[0],)

    # loop through all edges of the polygon
    for i in range(len(poly)-1):   # edge from poly[i] to poly[i+1]
        if ((poly[i][1] <= p[1] and poly[i+1][1] > p[1])        # an upward crossing
            or (poly[i][1] > p[1] and poly[i+1][1] <= p[1])):   # a downward crossing
            # compute the actual edge-ray intersect x-coordinate
            vt = (p[1] - poly[i][1]) / float(poly[i+1][1] - poly[i][1])
            if p[0] < poly[i][0] + vt * (poly[i+1][0] - poly[i][0]): # p[0] < intersect
                nCrossings += 1  # a valid crossing of y=p[1] right of p[0]

    return nCrossings % 2   # 0 if even (out), and 1 if odd (in)

# check if point p is outside of all rectangles
def isOutside(p, rectangles ):
    isOut = True
    for rectangle in rectangles:
        isOut = isOut and not countCrossings(p, rectangle)
    return isOut

# detect if <rectangle> collides with a recatngle in <rectangles>
# all rectangles are assumed to be a list of four vertices (mathutils.Vector) 
# in counter-clockwise order. The test is done using the separating axes 
# theorem (SAT)
def doCollideRect(rectangle, rectangles):
    if not rectangles:
        return False
    # find orthogonals of rectangle (no normaslization required)
    orth = (rectangle[1]-rectangle[0],rectangle[2]-rectangle[1])

    for rect in rectangles:
        orthogonals = orth + (rect[1]-rect[0],rect[2]-rect[1])
        sepAxisFound = False
        for o in orthogonals:
            # check projections onto <o>
            min1, max1 = float('+inf'), float('-inf')
            for v in rectangle:
                projection = o.dot(v)
                min1 = min(min1, projection)
                max1 = max(max1, projection)
            min2, max2 = float('+inf'), float('-inf')
            for v in rect:
                projection = o.dot(v)
                min2 = min(min2, projection)
                max2 = max(max2, projection)
            if not( max1 >= min2 and max2 >= min1 ):
                sepAxisFound = True
                break
        if not sepAxisFound:
            return True # we have a collision
 
    return False

# compute the x-value of the intersction of the segment p1-p2 with
# a line parallel to the x-axis at y
def xIntersection(p1, p2, y):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    x0 = p1[0]
    y0 = p1[1] + y
    return (x0+abs(y0/dy)*dx ) if dy else 0.


def fit_rectangles(verts, firstVertIndex, numVerts, holesInfo=None, unitVectors=None):
    """
    It accepts a simple description of the vertices of the footprint polygon,
    including those of evetual holes, and returns a list of polygon faces.

    The polygon is expected as a list of vertices in counterclockwise order. In a right-handed coordinate
    system, seen from top, the polygon is on the left of its contour. Holes are expected as lists of vertices
    in clockwise order. Seen from top, the polygon is on the left of the hole's contour.

    Arguments:
    ----------
    verts:              A list of vertices. Vertices that define the outer contour of the footprint polygon are
                        located in a continuous block of the verts list, starting at the index firstVertIndex.
                        Each vertex is an instance of mathutils.Vector with 3 coordinates x, y and z.

                        The outer contour of the footprint polygon contains numVerts vertices in counterclockwise
                        order, in its block in verts.

                        Vertices that define eventual holes are also located in verts. Every hole takes its continuous
                        block. The start index and the length of every hole block are described by the argument
                        holesInfo. See there.

    firstVertIndex: 	The first index of vertices of the polygon index in the verts list that defines the footprint polygon.

    numVerts:           The first index of the vertices in the verts list of the polygon, that defines the outer
                        contour of the footprint.

    holesInfo:          If the footprint polygon contains holes, their position and length in the verts list are
                        described by this argument. holesInfo is a list of tuples, one for every hole. The first
                        element in every tuple is the start index of the hole's vertices in verts and the second
                        element is the number of its vertices.

                        The default value of holesInfo is None, which means that there are no holes.

    unitVectors:        A Python list of unit vectors along the polygon edges (including holes if they are present).
                        These vectors are of type mathutils.Vector with three dimensions. The direction of the vectors
                        corresponds to order of the vertices in the polygon and its holes. The order of the unit
                        vectors in the unitVectors list corresponds to the order of vertices in the input Python list
                        verts.

                        The list unitVectors (if given) gets used inside polygonize() function instead of calculating
                        it once more. If this argument is None (its default value), the unit vectors get calculated
                        inside polygonize().

    Output:
    ------
    return:             A list of rectangles, every rectangle as list of vertices (mathutils.Vector) in 
                        counter-clockwise order.
    """
    # compute center of gravity of polygon
    center = sum(
        ( verts[i] for i in range(firstVertIndex, firstVertIndex+numVerts) ),
        mathutils.Vector((0., 0.))
    )/numVerts

    # create 2D edges as list and as contours for skeletonization by bpypolyskel
    lastUIndex = numVerts-1
    lastVertIndex = firstVertIndex + lastUIndex
    if unitVectors:
        edges2D = [
            Edge2(index, index+1, unitVectors[uIndex], verts, center)\
                for index, uIndex in zip( range(firstVertIndex, lastVertIndex), range(lastUIndex) )
        ]
        edges2D.append(Edge2(lastVertIndex, firstVertIndex, unitVectors[lastUIndex], verts, center))
    else:
        edges2D = [
            Edge2(index, index+1, None, verts, center) for index in range(firstVertIndex, lastVertIndex)
        ]
        edges2D.append(Edge2(lastVertIndex, firstVertIndex, None, verts, center))
    edgeContours = [edges2D.copy()]
    
    uIndex = numVerts
    if holesInfo:
        for firstVertIndexHole,numVertsHole in holesInfo:
            lastVertIndexHole = firstVertIndexHole + numVertsHole-1
            if unitVectors:
                lastUIndex = uIndex+numVertsHole-1
                holeEdges = [
                    Edge2(index, index+1, unitVectors[uIndex], verts, center)\
                    for index, uIndex in zip(range(firstVertIndexHole, lastVertIndexHole), range(uIndex, lastUIndex))
                ]
                holeEdges.append(Edge2(lastVertIndexHole, firstVertIndexHole, unitVectors[lastUIndex], verts, center))
            else:
                holeEdges = [
                    Edge2(index, index+1, None, verts, center) for index in range(firstVertIndexHole, lastVertIndexHole)
                ]
                holeEdges.append(Edge2(lastVertIndexHole, firstVertIndexHole, None, verts, center))
            edges2D.extend(holeEdges)
            edgeContours.append(holeEdges)
            uIndex += numVertsHole

	# compute skeleton
    skeleton = skeletonize(edgeContours)

    # extract skeleton nodes and heights
    s_nodes = [mathutils.Vector((arc.source[0], arc.source[1])) for arc in skeleton]
    s_heights = [arc.height for arc in skeleton]

    # find all skeleton edges that are not connected to the polygon
    skelEdges = []
    for index, arc in enumerate(skeleton):
        index1 = index
        for sink in arc.sinks:
            skelIndex = [index for index, arc in enumerate(skeleton) if arc.source==sink]
            if skelIndex:
                index2 = skelIndex[0]
                skelEdges.append((index1,index2))

    # find ridges (indices of roof edges) (egde slope < MAX_SLOPE)
    ridgeEdges = [(i1,i2) for i1,i2 in skelEdges if abs(s_heights[i1]-s_heights[i2])/(s_nodes[i1]-s_nodes[i2]).magnitude < MAX_SLOPE]

    # remove ridge vertices that have straight angles between their edges
    graph = {}
    for i1,i2 in ridgeEdges:
        if i1 not in graph:
            graph[i1] = []
        if i2 not in graph:
            graph[i2] = []
        graph[i1].append(i2)
        graph[i2].append(i1)

    for key, value in graph.items():
        if len(value)==2:
            r1 = (s_nodes[value[0]] - s_nodes[key]).normalized()
            r2 = (s_nodes[value[1]] - s_nodes[key]).normalized()
            if abs(r1.cross(r2)) < MAX_STRAIGHT:
                ridgeEdges = [ edge for edge in ridgeEdges if key not in edge]
                ridgeEdges.append(value)

    # sort ridge edges by descending minimum skeleton height
    sortRidges = np.argsort( [min(s_heights[i1],s_heights[i2]) for i1,i2 in ridgeEdges] )[::-1]

    # these lists will also contain rectangle vertices and edges
    # to detect intersecteions
    vertices = [e.p1 for e in edges2D]
    edges = [(edge.i1,edge.i2) for edge in edges2D]
    rectangles = []

    # now loop throught roof ridges
    for ridge in sortRidges:
        # find anchor and unit vector outside of existing rectangles
        # try first the nodes of the ridge and then the midpoint if no success
        i1, i2 = ridgeEdges[ridge]
        v1 = skeleton[i1].source
        v2 = skeleton[i2].source
        mid = v1 + (v2-v1)/2.
        anchor = None
        if isOutside(v1,rectangles):
            anchor = v1
            uVec = (v2-anchor).normalized()
        elif isOutside(v2,rectangles):
            anchor = v2
            uVec = (v1-anchor).normalized()
        elif isOutside(mid,rectangles):
            anchor = mid
            uVec = (v1-anchor).normalized()
        else:
            # ridge is assumed to be in a rectangle, try next one
            continue

        # shift and rotate all vertices so that anchor is at (0,0) and ridge is along x-axis
        rotFwd = mathutils.Matrix( [ (uVec[0], -uVec[1]), (uVec[1], uVec[0]) ] )
        rotVerts = [ (v-anchor) @ rotFwd for v in vertices ]

        # the distance of edges almost parallel to this ridge is given by the height
        # of the nodes in the skeleton. We take the minimal height and subtract a
        # safety margin of size MARGIN to get the horizontal dimension as long as possible 
        yDist = min(skeleton[i1].height,skeleton[i2].height) - MARGIN
        
        # get the x-values of all vertices in the range -yDist < y < yDist
        verticesX = [v[0] for v in rotVerts if abs(v[1]) < yDist ]

        # add the x-values of the intersections of lines parallel to the x-axis
        # at y=-yDist and y=yDist
        for i1,i2 in edges:
            if np.diff( np.sign( [rotVerts[i1][1]-yDist,rotVerts[i2][1]-yDist] ) ):
                verticesX.append( xIntersection(rotVerts[i1],rotVerts[i2],-yDist) )
            if np.diff( np.sign( [rotVerts[i1][1]+yDist,rotVerts[i2][1]+yDist] ) ):
                verticesX.append( xIntersection(rotVerts[i1],rotVerts[i2],yDist) )

        # find x-limits left and right of the anchor
        leftX  = max([x for x in verticesX if x < 0.])+ 0.1
        rightX = min([x for x in verticesX if x >= 0.]) - 0.1

        # discard rectangle if too small
        if (rightX-leftX) < MINDIM and 2*yDist < MINDIM:
            continue

        # define rectangle vertices and edges
        rectVerts = [
            mathutils.Vector((leftX, -yDist)),
            mathutils.Vector((rightX,-yDist)),
            mathutils.Vector((rightX, yDist)),
            mathutils.Vector((leftX,  yDist))
        ]
        N = len(edges)
        rectEdges = [ (N,N+1), (N+1,N+2), (N+2,N+3), (N+3,N) ]

        # rotate and shift this rectangle back to the original position
        rotRev = mathutils.Matrix( [ (uVec[0], uVec[1]), (-uVec[1], uVec[0]) ] )
        rectVerts = [ (v @ rotRev)+anchor for v in rectVerts ]

        # check for intersection of this rectangle with existing rectangles,
        # discard it if there are intersections
        if doCollideRect(rectVerts, rectangles):
            continue

        # we got a rectagle :)
        vertices.extend(rectVerts)
        edges.extend(rectEdges)
        rectangles.append(rectVerts)

    # return a Python tuple of rectangles, every rectangle as list of vertices in counter-clockwise order
    rectList = tuple(
        tuple( (v + center) for v in rectangle ) for rectangle in rectangles
    )
    return rectList