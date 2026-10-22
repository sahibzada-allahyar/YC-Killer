import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { ThreeCanvas } from "@remotion/three";
import * as THREE from "three";
import { loadFont as loadSerif } from "@remotion/google-fonts/InstrumentSerif";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

const SERIF = loadSerif().fontFamily;
const MONO = loadMono().fontFamily;

export const HERO_FRAMES = 180; // 6 s at 30 fps, seamless loop
const N = 300;
const BG = "#101314", CREAM = new THREE.Color("#E8E4D8"), BLUE = new THREE.Color("#6B95F0"),
  RED = new THREE.Color("#E8553B"), DIM = new THREE.Color("#2E3538");

// deterministic: which steps are "fake proofs" the gate knocks out
const FAKE = (() => {
  let a = 7;
  const r = () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  return new Set(Array.from({ length: 22 }, () => Math.floor(r() * N)));
})();

const Knot: React.FC = () => {
  const f = useCurrentFrame();
  const u = f / HERO_FRAMES; // pulse head, 0..1
  const curve = useMemo(() => {
    const pts = Array.from({ length: N }, (_, i) => {
      const t = (i / N) * Math.PI * 2, p = 2, q = 3;
      const r = 2.1 + 0.85 * Math.cos(q * t);
      return new THREE.Vector3(r * Math.cos(p * t), r * Math.sin(p * t), 0.95 * Math.sin(q * t));
    });
    return pts;
  }, []);
  const spin = (f / HERO_FRAMES) * Math.PI * 2;
  return (
    <group rotation={[0.62, 0.18, 0]} scale={0.97}>
    <group rotation={[0, 0, spin / 3]}>
      {curve.map((pt, i) => {
        const d = (((u - i / N) % 1) + 1) % 1; // how long ago the pulse passed this step
        const fake = FAKE.has(i);
        let color = DIM.clone(), s = 0.055, halo = 0;
        if (d < 0.7) {
          const k = Math.pow(1 - d / 0.7, 1.6);
          if (fake) {
            const flash = Math.max(0, 1 - d / 0.1);
            color = RED.clone().lerp(DIM, Math.min(1, d / 0.7) * 0.6);
            s = 0.07 + 0.16 * flash;
            halo = 0.15 + 0.7 * flash;
            if (d > 0.1) s = 0.05; // rejected: stays red, never turns blue
          } else {
            color = DIM.clone().lerp(BLUE, 0.35 + 0.65 * k).lerp(CREAM, d < 0.015 ? 0.8 : 0);
            s = 0.055 + 0.05 * k;
            halo = k * 0.55;
          }
        }
        return (
          <group key={i} position={pt}>
            <mesh scale={s}>
              <icosahedronGeometry args={[1, 1]} />
              <meshBasicMaterial color={color} />
            </mesh>
            {halo > 0.02 ? (
              <mesh scale={s * 3.2}>
                <sphereGeometry args={[1, 12, 12]} />
                <meshBasicMaterial color={fake ? RED : BLUE} transparent opacity={0.18 * halo} depthWrite={false}
                  blending={THREE.AdditiveBlending} />
              </mesh>
            ) : null}
          </group>
        );
      })}
      {/* the pulse head */}
      {(() => {
        const i = Math.floor(u * N) % N;
        return (
          <mesh position={curve[i]} scale={0.16}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshBasicMaterial color={CREAM} transparent opacity={0.9} blending={THREE.AdditiveBlending} />
          </mesh>
        );
      })()}
    </group>
    </group>
  );
};

const Dust: React.FC = () => {
  const f = useCurrentFrame();
  const pts = useMemo(() => Array.from({ length: 160 }, (_, i) => {
    const a = i * 2.399963, r = 5 + (i % 13) * 0.45;
    return [Math.cos(a) * r, ((i * 37) % 100) / 100 * 7 - 3.5, Math.sin(a) * r - 3] as const;
  }), []);
  const rot = (f / HERO_FRAMES) * Math.PI * 2 * 0.25;
  return (
    <group rotation={[0, rot, 0]}>
      {pts.map(([x, y, z], i) => (
        <mesh key={i} position={[x, y, z]} scale={0.02 + (i % 5) * 0.006}>
          <sphereGeometry args={[1, 6, 6]} />
          <meshBasicMaterial color={CREAM} transparent opacity={0.25} />
        </mesh>
      ))}
    </group>
  );
};

export const Hero: React.FC = () => {
  const { width, height } = useVideoConfig();
  const f = useCurrentFrame();
  const proved = Math.floor((f / HERO_FRAMES) * (N - FAKE.size));
  const rejected = Math.floor((f / HERO_FRAMES) * FAKE.size);
  return (
    <AbsoluteFill style={{ background: `radial-gradient(90% 120% at 68% 50%, #18211f 0%, ${BG} 60%, #0b0d0e 100%)` }}>
      <AbsoluteFill style={{ left: width * 0.28 }}>
        <ThreeCanvas width={width * 0.72} height={height} camera={{ position: [0, 0, 9.2], fov: 42 }}>
          <Knot />
          <Dust />
        </ThreeCanvas>
      </AbsoluteFill>
      <div style={{ position: "absolute", left: 64, top: 0, bottom: 0, display: "flex", flexDirection: "column",
        justifyContent: "center", width: 560 }}>
        <div style={{ fontFamily: MONO, fontSize: 15, letterSpacing: 3, color: "#6B95F0" }}>∎ LEAN · DAFNY · RSI</div>
        <div style={{ fontFamily: SERIF, fontSize: 76, lineHeight: 0.98, color: "#E8E4D8", marginTop: 14 }}>
          Millennium<br />Harness
        </div>
        <div style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 30, color: "#C9C3B3", marginTop: 18 }}>
          Done means proved.
        </div>
        <div style={{ fontFamily: MONO, fontSize: 15, color: "#8B9296", marginTop: 26, lineHeight: 1.7 }}>
          <span style={{ color: "#6B95F0" }}>✓ {String(proved).padStart(3, "0")} kernel-checked</span>
          {"   "}
          <span style={{ color: "#E8553B" }}>✗ {String(rejected).padStart(2, "0")} fakes rejected</span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
