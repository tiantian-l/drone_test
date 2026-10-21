"""Synchronized diagnostic projections; no contact-point inference."""
import numpy as np


def objects(geometry, row):
    for body, spec in zip(geometry['static_body_ids'], geometry['static_boxes']):
        x,y,wx,wy,h=spec
        yield body,x,y,wx,wy,h,False
    for body, spec in zip(geometry['dynamic_body_ids'], row['dynamic_cylinders']):
        x,y,r,h=spec[:4]
        yield body,x,y,2*r,2*r,h,True


def terminal_ids(episode, geometry):
    return {x['body_id'] for x in episode['terminal']['nearby_obstacles']
            if x['clearance'] < geometry['collision_distance']}


def draw_views(top, side, geometry, rows, states, times, i, begin, highlighted, color):
    from matplotlib.patches import Rectangle, Circle
    top.clear();side.clear()
    bounds=np.asarray(geometry['bounds']);pos=states[i,:3]
    current=list(objects(geometry,rows[i]))
    for body,x,y,wx,wy,h,dynamic in current:
        face='crimson' if body in highlighted else color(h)
        patch=Circle((x,y),wx/2) if dynamic else Rectangle((x-wx/2,y-wy/2),wx,wy)
        patch.set(facecolor=face,edgecolor='#455466',alpha=.7);top.add_patch(patch)
    top.plot(states[begin:i+1,0],states[begin:i+1,1],color='#006bcb',lw=1.5)
    top.scatter(*pos[:2],s=28,color='#006bcb',zorder=5)
    top.scatter(*geometry['goal'][:2],marker='*',s=55,color='green')
    if i==len(rows)-1:
        top.scatter(*pos[:2],marker='x',s=90,color='crimson',zorder=6)
    top.set(xlim=(min(bounds[0,0],states[:,0].min())-.3,max(bounds[0,1],states[:,0].max())+.3),
            ylim=(min(bounds[1,0],states[:,1].min())-.3,max(bounds[1,1],states[:,1].max())+.3),
            xlabel='x (m)',ylabel='y (m)',title='Synchronized top view')
    top.set_aspect('equal',adjustable='box')
    draw_closeup(side, geometry, rows, states, times, i, begin, highlighted, color)
    top.grid(alpha=.15);top.tick_params(labelsize=7)
    top.xaxis.label.set_size(8);top.yaxis.label.set_size(8);top.title.set_size(9)


def closeup_limits(terminal):
    """Fixed metric volume around the terminal drone position, independent of frame."""
    terminal=np.asarray(terminal)
    return np.array([[terminal[0]-2,terminal[0]+2],
                     [terminal[1]-2,terminal[1]+2],
                     [max(0,terminal[2]-1.5),max(0,terminal[2]-1.5)+3]])


def draw_closeup(ax, geometry, rows, states, times, i, begin, highlighted, color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    terminal=states[-1,:3];pos=states[i,:3]
    limits=closeup_limits(terminal)
    focus=next((o for o in objects(geometry,rows[-1]) if o[0] in highlighted),None)
    bearing=np.array(focus[1:3])-terminal[:2] if focus else terminal[:2]-states[max(0,len(states)-31),:2]
    if np.linalg.norm(bearing)<1e-8:bearing=np.array([1.,0.])
    # Side-on oblique camera, avoiding looking straight through the target obstacle.
    azimuth=np.degrees(np.arctan2(bearing[1],bearing[0]))+110
    ax.view_init(elev=28,azim=azimuth)
    ax.set_proj_type('ortho');ax.set_box_aspect((4,4,3))
    ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2])
    ax.set_xlabel('x (m)',fontsize=6,labelpad=-5)
    ax.set_ylabel('y (m)',fontsize=6,labelpad=-5)
    ax.set_zlabel('z (m)',fontsize=6,labelpad=-5)
    ax.tick_params(labelsize=6,pad=0)
    ax.set_title('Terminal-region 3D close-up (fixed camera)',fontsize=9,pad=0)
    if times[i]<times[-1]-2:
        ax.text2D(.1,.45,'Close-up shown during final 2 seconds',transform=ax.transAxes,fontsize=8)
        return
    for body,x,y,wx,wy,h,dynamic in objects(geometry,rows[i]):
        if x+wx/2<limits[0,0] or x-wx/2>limits[0,1] or y+wy/2<limits[1,0] or y-wy/2>limits[1,1]:continue
        z0=max(0,limits[2,0]);z1=min(h,limits[2,1])
        if z1<=z0:continue
        if dynamic:
            angles=np.arange(25)*2*np.pi/24
            bottom=np.column_stack((x+wx/2*np.cos(angles),y+wy/2*np.sin(angles),np.full(25,z0)))
            topface=bottom.copy();topface[:,2]=z1
            faces=[topface]+[np.array([bottom[k],bottom[k+1],topface[k+1],topface[k]]) for k in range(24)]
        else:
            xl,xh=max(x-wx/2,limits[0,0]),min(x+wx/2,limits[0,1])
            yl,yh=max(y-wy/2,limits[1,0]),min(y+wy/2,limits[1,1])
            vertices=np.array([[xx,yy,zz] for zz in [z0,z1] for xx,yy in [(xl,yl),(xh,yl),(xh,yh),(xl,yh)]])
            faces=[vertices[list(ids)] for ids in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]
        ax.add_collection3d(Poly3DCollection(faces,facecolors='crimson' if body in highlighted else color(h),
                                          edgecolors='#455466',linewidths=.5,alpha=.3 if body in highlighted else .12))
        if body in highlighted:
            ax.text(np.clip(x,*limits[0]),np.clip(y,*limits[1]),z1,f'ID {body} | h={h:.2f}m',fontsize=7)
    # NaNs break the trail at volume exits rather than connecting across invisible spans.
    track=states[begin:i+1,:3].copy()
    inside=np.all((track>=limits[:,0]) & (track<=limits[:,1]),axis=1)
    track[~inside]=np.nan
    ax.plot(*track.T,color='#006bcb',lw=2)
    if np.all((pos>=limits[:,0]) & (pos<=limits[:,1])):
        ax.scatter(*pos,color='#006bcb',s=35,depthshade=False)
        if i==len(rows)-1:ax.scatter(*pos,marker='x',color='crimson',s=90,depthshade=False)
    # Actual flight boundaries, not inferred contact points.
    bounds=np.asarray(geometry['bounds'])
    for axis in [0,1]:
        for edge in bounds[axis]:
            if limits[axis,0]<=edge<=limits[axis,1]:
                other=1-axis
                line=np.zeros((2,3));line[:,axis]=edge;line[:,other]=limits[other];line[:,2]=pos[2]
                ax.plot(*line.T,color='red',ls='--',lw=1.5)
    near=rows[i]['nearby_obstacles']
    relevant=[o['clearance'] for o in near if o['body_id'] in highlighted] if highlighted else []
    clearance=f'{min(relevant):.3f} m' if relevant else 'not measured'
    ax.text2D(0,-.08,f'Drone z={pos[2]:.2f} m | target clearance: {clearance}\nCropped volume; marker is drone center, not contact point',
              transform=ax.transAxes,fontsize=7)
