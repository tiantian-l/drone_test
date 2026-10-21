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
    # A fixed vertical projection aligned to the terminal drone-obstacle bearing.
    # This is a projection, not a cross-section or nearest-surface measurement.
    terminal=states[-1,:3]
    focus=next((o for o in objects(geometry,rows[-1]) if o[0] in highlighted),None)
    direction=np.array([focus[1]-terminal[0],focus[2]-terminal[1]]) if focus else np.array([1.,0.])
    if np.linalg.norm(direction)<1e-8:direction=np.array([1.,0.])
    direction/=np.linalg.norm(direction)
    span=2.5
    if focus:span=max(span,np.linalg.norm(np.array(focus[1:3])-terminal[:2])+max(focus[3:5])/2+.5)
    for body,x,y,wx,wy,h,dynamic in current:
        if highlighted and body not in highlighted:continue
        center=float((np.array([x,y])-terminal[:2])@direction)
        half=wx/2 if dynamic else abs(direction[0])*wx/2+abs(direction[1])*wy/2
        if abs(center)>span+half:continue
        side.add_patch(Rectangle((center-half,0),2*half,h,facecolor='crimson' if body in highlighted else color(h),alpha=.5,edgecolor='#455466'))
        side.text(center,h+.08,f'ID {body}: {h:.2f} m',ha='center',fontsize=7)
    along=(states[begin:i+1,:2]-terminal[:2])@direction
    side.plot(along,states[begin:i+1,2],color='#006bcb')
    side.scatter(float((pos[:2]-terminal[:2])@direction),pos[2],s=35,color='#006bcb',zorder=5)
    if i==len(rows)-1:side.scatter(0,terminal[2],marker='x',s=90,color='crimson',zorder=6)
    for z in bounds[2]:side.axhline(z,color='#777',ls='--',lw=.8)
    side.set(xlim=(-span,span),ylim=(min(0,states[:,2].min()-.3),max(bounds[2,1],max((o[5] for o in current if o[0] in highlighted),default=0),states[:,2].max())+.5),
             xlabel='Projected distance from terminal position (m)',ylabel='Height (m)',
             title='Local side projection (not a contact section)')
    for ax in [top,side]:ax.grid(alpha=.15);ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.title.set_size(9)
