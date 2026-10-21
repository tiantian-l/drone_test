"""Render recorded UAV geometry as a fixed-camera 3D MP4; no simulator/GPU needed."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from analyze_diagnostics import resolve_directory, load_episode


def box_faces(x, y, wx, wy, height):
    v = np.array([[x+sx*wx/2, y+sy*wy/2, z] for z in [0, height]
                  for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    return [v[list(ids)] for ids in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]


def cylinder_faces(spec, n=16):
    x,y,r,h = spec[:4]
    a = np.arange(n)*2*np.pi/n
    bottom = np.column_stack((x+r*np.cos(a), y+r*np.sin(a), np.zeros(n)))
    top = bottom.copy(); top[:,2] = h
    return [top] + [np.array([bottom[i], bottom[(i+1)%n], top[(i+1)%n], top[i]]) for i in range(n)]


def camera(geometry, rows, elevation=52, height_ratio=.8, azimuth=None):
    """Physical camera position, auto-fitted projection, and complete scene bounds."""
    bounds = np.asarray(geometry['bounds'], float)
    low = np.minimum(bounds[:,0], np.min([r['state'][:3] for r in rows], axis=0))
    high = np.maximum(bounds[:,1], np.max([r['state'][:3] for r in rows], axis=0))
    points = [geometry['start'], geometry['goal']]
    for b in geometry['static_boxes']:
        points.extend(np.concatenate(box_faces(*b)))
    for row in rows:
        for spec in row['dynamic_cylinders']:
            x,y,r,h = spec[:4]
            points.extend([[x-r,y-r,0],[x+r,y+r,h]])
    low = np.minimum(low,np.min(points,axis=0)); high=np.maximum(high,np.max(points,axis=0))
    length = float(max(high[:2]-low[:2]))
    target = np.mean(bounds, axis=1)
    start_edge = None
    if azimuth is None:
        # Use the nearest flight-region edge, not the randomized start-to-goal
        # direction. Keep the camera on that edge midpoint's outward normal.
        start = np.asarray(geometry['start'])
        distances = [abs(start[0]-bounds[0,0]), abs(start[0]-bounds[0,1]),
                     abs(start[1]-bounds[1,0]), abs(start[1]-bounds[1,1])]
        edge = int(np.argmin(distances))
        start_edge = ['x_min', 'x_max', 'y_min', 'y_max'][edge]
        azimuth = [180., 0., -90., 90.][edge]
    az=math.radians(azimuth); el=math.radians(elevation)
    camera_z=max(height_ratio*length,target[2]+.1*length)
    distance=(camera_z-target[2])/math.tan(el)
    if start_edge is not None:
        axis = 0 if start_edge.startswith('x') else 1
        distance = max(distance, (bounds[axis,1]-bounds[axis,0])/2 + .3*length)
        camera_z = target[2] + distance*math.tan(el)
    offset=np.array([distance*math.cos(az),distance*math.sin(az),camera_z-target[2]])
    eye=target+offset
    forward=(target-eye)/np.linalg.norm(target-eye)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
    # Move backward along the same sight line if any scene corner is behind the camera.
    depth=(corners-eye)@forward
    if depth.min() < .15*length:
        eye-=forward*(.15*length-depth.min())
    def project(vertices):
        rel=np.asarray(vertices)-eye
        depth=rel@forward
        return np.column_stack((rel@right/depth,rel@up/depth)),depth
    # Avoid a near-corner forcing an extreme wide-angle, tiny central scene.
    for _ in range(100):
        projected,_=project(corners)
        extent=max(float(np.max(np.abs(projected[:,1]))),
                   float(np.max(np.abs(projected[:,0])))/(16/9))*1.12
        if extent <= math.tan(math.radians(37.5)):
            break
        eye -= forward * (.1*length)
    meta=dict(position=eye.tolist(),target=target.tolist(),elevation_deg=elevation,
        azimuth_deg=azimuth,start_edge=start_edge,scene_length=length,height_ratio_requested=height_ratio,
        vertical_fov_deg=math.degrees(2*math.atan(extent)))
    return project, extent, meta


def frame_indices(times, fps, speed, start, end, hold):
    sample=np.arange(start,end, speed/fps)
    sample=np.append(sample,end)
    indices=np.searchsorted(times,sample,side='right')-1
    indices=np.clip(indices,0,len(times)-1)
    return np.concatenate((indices,np.repeat(indices[-1],round(hold*fps))))


def render(directory, seed, output, fps=20, speed=1., start=0., end=None,
           elevation=52., height_ratio=.8, azimuth=None, tail=1., hold=1., width=1280, layout='multi', slow_last=2., slow_factor=.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.animation import FFMpegWriter, writers
    if not writers.is_available('ffmpeg'):
        raise RuntimeError('FFmpeg is required for MP4 output; install ffmpeg in the active environment.')
    if not (fps>0 and speed>0 and height_ratio>0 and 0<elevation<90 and tail>=0 and hold>=0 and width>=320):
        raise ValueError('Invalid camera/playback settings.')
    if layout not in ('single','multi') or slow_last<0 or not 0<slow_factor<=1:
        raise ValueError('Invalid layout or slowdown settings')
    directory=resolve_directory(directory)
    matches=list(directory.glob(f'*_seed_{seed}.episode.json'))
    if len(matches)!=1: raise ValueError(f'Expected one recorded episode for seed {seed}.')
    episode,geometry,rows=load_episode(matches[0])
    times=np.array([r['time_s'] for r in rows]); states=np.array([r['state'] for r in rows])
    end=times[-1] if end is None else min(end,times[-1])
    if not 0<=start<end: raise ValueError('Require 0 <= start < end <= episode duration.')
    output=Path(output)
    if output.suffix.lower()!='.mp4': raise ValueError('Output must end in .mp4')
    if output.exists() or output.with_suffix('.camera.json').exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    project,extent,meta=camera(geometry,rows,elevation,height_ratio,azimuth)
    from replay_views import objects, terminal_ids, draw_views
    highlighted=terminal_ids(episode,geometry)
    max_height=max([b[4] for b in geometry['static_boxes']]+[b[3] for b in geometry['dynamic_cylinders']]+[1.])
    norm=matplotlib.colors.Normalize(0,max_height)
    height_color=lambda h: plt.get_cmap('viridis')(norm(h))
    static=[(f,'crimson' if body in highlighted else height_color(b[4]))
            for body,b in zip(geometry['static_body_ids'],geometry['static_boxes']) for f in box_faces(*b)]
    bounds=geometry['bounds']
    ground=np.array([[bounds[0][0],bounds[1][0],0],[bounds[0][1],bounds[1][0],0],
                     [bounds[0][1],bounds[1][1],0],[bounds[0][0],bounds[1][1],0]])
    width=int(width)//2*2;height=round(width*9/16)//2*2
    fig=plt.figure(figsize=(width/100,height/100),dpi=100)
    if layout=='multi':
        ax=fig.add_axes([.01,.1,.59,.79])
        top=fig.add_axes([.67,.57,.30,.34]);side=fig.add_axes([.64,.11,.35,.34],projection='3d')
    else:
        ax=fig.add_axes([0,.07,1,.82])
    bar=fig.add_axes([.12,.095,.35,.015])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=bar,orientation='horizontal',label='Obstacle height (m); red = terminal threshold obstacle')
    header=fig.text(.02,.98,'',va='top',fontsize=9)
    fig.text(.02,.012,'Recorded geometry | transparent / x-ray overlays | enlarged drone | terminal X marks drone position, NOT contact point',fontsize=7)
    labels=[k for k in ['success','collision','crash','timeout'] if episode['outcome'][k]]
    labels += [k for k,v in episode['terminal']['events'].items() if v]
    cutoff=max(start,float(times[-1])-slow_last)
    slowed=not episode['outcome']['success'] and slow_last>0 and end>cutoff
    if slowed:
        first=frame_indices(times,fps,speed,start,cutoff,0)[:-1] if cutoff>start else np.array([],dtype=int)
        indices=np.concatenate((first,frame_indices(times,fps,speed*slow_factor,max(start,cutoff),end,hold)))
    else:
        indices=frame_indices(times,fps,speed,start,end,hold)
    writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    temporary=output.with_name(output.stem+'.partial.mp4')
    if temporary.exists(): raise FileExistsError(temporary)
    try:
        with writer.saving(fig,str(temporary),100):
            for i in indices:
                ax.clear();ax.set(xlim=(-extent*16/9,extent*16/9),ylim=(-extent,extent),aspect='equal');ax.axis('off')
                ax.set_facecolor('#f5f8fc')
                floor,_=project(ground);ax.add_collection(PolyCollection([floor],facecolors='#e8eef4',edgecolors='#9aabba'))
                faces=list(static)
                for body,obstacle in zip(geometry['dynamic_body_ids'],rows[i]['dynamic_cylinders']):
                    faces.extend((f,'crimson' if body in highlighted else height_color(obstacle[3])) for f in cylinder_faces(obstacle))
                projected=[(project(f),color) for f,color in faces]
                projected.sort(key=lambda item:float(item[0][1].mean()),reverse=True)
                for (xy,_),color in projected:
                    ax.add_collection(PolyCollection([xy],facecolors=color,edgecolors='#596777',linewidths=.3,alpha=.45))
                # Deliberate x-ray overlay for diagnosis: never represents sensor visibility.
                begin=np.searchsorted(times,times[i]-tail)
                track,_=project(states[begin:i+1,:3]);ax.plot(track[:,0],track[:,1],color='#086bd6',lw=2)
                for point,marker,color in [(geometry['start'],'o','green'),(geometry['goal'],'*','green')]:
                    xy,_=project([point]);ax.scatter(*xy[0],marker=marker,color=color,s=65)
                pos=states[i,:3];roll,pitch,yaw=states[i,7:10]
                cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
                rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
                arm=meta['scene_length']*.012
                for body in [np.array([[-arm,0,0],[arm,0,0]]),np.array([[0,-arm,0],[0,arm,0]])]:
                    xy,_=project(body@rotation.T+pos);ax.plot(xy[:,0],xy[:,1],color='#0056ae',lw=3)
                xy,_=project([pos]);ax.scatter(*xy[0],s=22,color='red' if i==len(rows)-1 and not episode['outcome']['success'] else '#0056ae')
                near=rows[i]['nearby_obstacles']
                clearance=f"{min(x['clearance'] for x in near):.3f} m" if near else f">= {rows[i]['clearance_query_radius']:.2f} m (query limit)"
                status=' + '.join(labels) if i==len(rows)-1 else 'in flight'
                actual_speed=speed*slow_factor if slowed and times[i]>=cutoff else speed
                header.set_text(f'Seed {seed} | t={times[i]:.2f}s | playback {actual_speed:g}x | height {pos[2]:.2f}m | speed {np.linalg.norm(states[i,10:13]):.2f}m/s\nClearance: {clearance} | {status}')
                if i==len(rows)-1:
                    ax.scatter(*xy[0],marker='x',s=100,color='crimson')
                if layout=='multi':
                    draw_views(top,side,geometry,rows,states,times,i,begin,highlighted,height_color)
                writer.grab_frame()
        temporary.replace(output)
        meta.update(seed=seed,fps=fps,playback_speed=speed,frames=len(indices),start_s=start,end_s=float(end),
                    layout=layout,slow_last=slow_last,slow_factor=slow_factor,terminal_obstacle_ids=sorted(highlighted),
                    source=str(directory.resolve()),sample_method='previous recorded state (no interpolation)')
        output.with_suffix('.camera.json').write_text(json.dumps(meta,indent=2)+'\n')
    finally:
        plt.close(fig)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--fps',type=int,default=20);p.add_argument('--speed',type=float,default=1)
    p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float)
    p.add_argument('--elevation',type=float,default=52);p.add_argument('--height-ratio',type=float,default=.8)
    p.add_argument('--azimuth',type=float);p.add_argument('--tail',type=float,default=1)
    p.add_argument('--hold',type=float,default=1);p.add_argument('--width',type=int,default=1280)
    p.add_argument('--layout',choices=['single','multi'],default='multi')
    p.add_argument('--slow-last',type=float,default=2)
    p.add_argument('--slow-factor',type=float,default=.5)
    a=p.parse_args()
    print(render(a.input,a.seed,a.output or a.input/'videos'/f'seed_{a.seed}.mp4',
        a.fps,a.speed,a.start,a.end,a.elevation,a.height_ratio,a.azimuth,a.tail,a.hold,a.width,a.layout,a.slow_last,a.slow_factor))


if __name__=='__main__':main()
