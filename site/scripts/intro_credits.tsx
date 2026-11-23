import React from 'react';
import {AbsoluteFill, Composition, Easing, interpolate, registerRoot, useCurrentFrame, useVideoConfig} from 'remotion';

const palette = ['#ff555c', '#ffcc00', '#33ff33', '#00bccc', '#9933ff'];

function Credits() {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const opacity = interpolate(frame, [0, 0.9*fps, durationInFrames-0.65*fps, durationInFrames-1],
    [0, 1, 1, 0], {extrapolateLeft:'clamp', extrapolateRight:'clamp', easing:Easing.inOut(Easing.cubic)});
  const reveal = interpolate(frame, [0.25*fps, 1.25*fps], [0, 1], {
    extrapolateLeft:'clamp', extrapolateRight:'clamp', easing:Easing.out(Easing.cubic),
  });

  return <AbsoluteFill style={{background:'#000',color:'#f5f1ff',fontFamily:'Arial, sans-serif'}}>
    <div style={{position:'absolute',left:0,right:0,top:225,textAlign:'center',opacity}}>
      <div style={{display:'flex',justifyContent:'center',gap:7,marginBottom:25,scale:reveal+' 1'}}>
        {palette.map(color=><div key={color} style={{width:22,height:3,background:color,borderRadius:3}}/>)}
      </div>
      <div style={{fontSize:74,fontWeight:700,letterSpacing:-3}}>ChromoWorld</div>
      <div style={{marginTop:14,color:'#cbc5d4',fontSize:28}}>Chromosome identity in motion</div>
    </div>
    <div style={{position:'absolute',left:0,right:0,top:590,textAlign:'center',fontSize:20,color:'#b6afc1',opacity}}>
      “A Kind Of Hope” by Scott Buckley · CC BY 4.0 · scottbuckley.com.au
    </div>
  </AbsoluteFill>;
}

registerRoot(()=><Composition id="ChromoWorldCredits" component={Credits}
  durationInFrames={120} fps={24} width={1280} height={720}/>);
