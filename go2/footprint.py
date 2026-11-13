"""Collision distances for a translating rectangular robot footprint."""
import math

import numpy as np


def swept_distances(points, ux, uy, half_length, half_width):
    """Travel until each body-frame point enters the moving rectangle.

    Intersect the x/y time intervals (ray versus axis-aligned rectangle).
    Infinity means the point never enters the swept footprint. Points already
    inside return zero. This avoids the false corners of a travel-aligned
    bounding box when a holonomic robot moves diagonally.
    """
    if not all(math.isfinite(v) for v in (ux,uy,half_length,half_width)):
        raise ValueError('footprint and direction must be finite')
    if half_length<=0 or half_width<=0:
        raise ValueError('footprint half-dimensions must be positive')
    points=np.asarray(points,dtype=float)
    if not points.size:
        return np.empty(0)
    if points.ndim!=2 or points.shape[1]!=2:
        raise ValueError('points must have shape (N,2)')
    enter=np.full(len(points),-math.inf)
    leave=np.full(len(points),math.inf)
    for coordinate,velocity,extent in ((points[:,0],ux,half_length),(points[:,1],uy,half_width)):
        if abs(velocity)<1e-12:
            outside=abs(coordinate)>extent
            enter[outside],leave[outside]=math.inf,-math.inf
        else:
            a,b=(coordinate-extent)/velocity,(coordinate+extent)/velocity
            enter=np.maximum(enter,np.minimum(a,b))
            leave=np.minimum(leave,np.maximum(a,b))
    collision=(leave>=np.maximum(enter,0.)) & np.isfinite(points).all(axis=1)
    return np.where(collision,np.maximum(enter,0.),math.inf)
