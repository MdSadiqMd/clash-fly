import { useEffect, useRef } from "react";
import * as THREE from "three";

type Somas = { classes: string[]; xyz: number[][]; cls: number[] };

export const CLASS_COLOR: Record<string, number> = {
	olfactory: 0x4fd1c5,
	ALPN: 0x63b3ed,
	Kenyon_Cell: 0xf6e05e,
	MBON: 0xf687b3,
	DAN: 0xfc8181,
	APL: 0xb794f4,
};
const REST = 0x2f6f9e;

/** Soft round sprite so each soma renders as a glowing dot instead of a square. */
function makeSprite(): THREE.Texture {
	const c = document.createElement("canvas");
	c.width = c.height = 64;
	const g = c.getContext("2d");
	if (g) {
		const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
		grad.addColorStop(0, "rgba(255,255,255,1)");
		grad.addColorStop(0.35, "rgba(255,255,255,0.55)");
		grad.addColorStop(1, "rgba(255,255,255,0)");
		g.fillStyle = grad;
		g.fillRect(0, 0, 64, 64);
	}
	const t = new THREE.CanvasTexture(c);
	t.colorSpace = THREE.SRGBColorSpace;
	return t;
}

/**
 * FlyWire soma point cloud, fixed dorsal view (no rotation). Resting somata are dim blue;
 * `activeKc` lights those Kenyon cells, `pulse` lights a whole class once. Every lit soma
 * decays back over ~0.8 s so a decision reads as a wave: ORN -> PN -> KC -> MBON.
 */
export function SomaCloud({
	activeKc,
	pulse,
}: {
	activeKc: number[];
	pulse: { cls: string; color: number; key: number } | null;
}) {
	const host = useRef<HTMLDivElement>(null);
	const api = useRef<{
		light: (idx: number[], hex: number, strength?: number) => void;
		byClass: Record<string, number[]>;
	} | null>(null);

	useEffect(() => {
		const el = host.current;
		if (!el) return;
		const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(el.clientWidth, el.clientHeight);
		el.appendChild(renderer.domElement);
		const scene = new THREE.Scene();
		const camera = new THREE.PerspectiveCamera(
			36,
			el.clientWidth / el.clientHeight,
			1,
			5000,
		);
		let raf = 0;
		let disposed = false;
		const flash = new Map<number, number>();
		const byClass: Record<string, number[]> = {};
		let frameCloud = () => {};

		fetch("/somas.json")
			.then((r) => r.json())
			.then((data: Somas) => {
				if (disposed) return;
				const n = data.xyz.length;
				const pos = new Float32Array(n * 3);
				const col = new Float32Array(n * 3);
				const size = new Float32Array(n);
				const mean = [0, 1, 2].map(
					(k) => data.xyz.reduce((s, p) => s + p[k], 0) / n,
				);
				const names = data.cls.map((i) => data.classes[i]);
				const baseColor = (i: number) => {
					const hex = CLASS_COLOR[names[i]];
					return new THREE.Color(hex ?? REST).multiplyScalar(hex ? 0.5 : 0.32);
				};
				const ax = new Float32Array(n);
				const ay = new Float32Array(n);
				for (let i = 0; i < n; i++) {
					const p = data.xyz[i];
					const x = (p[0] - mean[0]) / 400;
					const y = -(p[1] - mean[1]) / 400;
					const z = (p[2] - mean[2]) / 400;
					pos.set([x, y, z], 3 * i);
					ax[i] = Math.abs(x);
					ay[i] = Math.abs(y);
					const c = baseColor(i);
					col.set([c.r, c.g, c.b], 3 * i);
					size[i] = CLASS_COLOR[names[i]] ? 1.0 : 0.7;
					byClass[names[i]] ??= [];
					byClass[names[i]].push(i);
				}
				const pct = (a: Float32Array) =>
					Float32Array.from(a).sort()[Math.floor(0.97 * (n - 1))] || 1;
				const hw = pct(ax);
				const hh = pct(ay);
				frameCloud = () => {
					const aspect = el.clientWidth / el.clientHeight;
					const tanHalf = Math.tan((camera.fov * Math.PI) / 360);
					camera.position.set(
						0,
						0,
						Math.max(hw / (0.88 * aspect * tanHalf), hh / (0.88 * tanHalf)),
					);
					camera.aspect = aspect;
					camera.updateProjectionMatrix();
				};
				frameCloud();
				const geo = new THREE.BufferGeometry();
				geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
				geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
				const material = new THREE.PointsMaterial({
					size: 4.5,
					map: makeSprite(),
					vertexColors: true,
					blending: THREE.AdditiveBlending,
					depthWrite: false,
					transparent: true,
					opacity: 0.85,
					sizeAttenuation: true,
				});
				const points = new THREE.Points(geo, material);
				points.rotation.x = 0.28; // slight dorsal tilt; static, no spin
				scene.add(points);
				api.current = {
					byClass,
					light: (idx, hex, strength = 1) => {
						const c = new THREE.Color(hex).multiplyScalar(1.6 * strength);
						for (const i of idx) {
							col.set([c.r, c.g, c.b], 3 * i);
							flash.set(i, 1);
						}
						geo.attributes.color.needsUpdate = true;
					},
				};
				let last = performance.now();
				const tick = (now: number) => {
					if (disposed) return;
					raf = requestAnimationFrame(tick);
					const dt = Math.min((now - last) / 1000, 0.1);
					last = now;
					if (flash.size > 0) {
						for (const [i, v] of flash) {
							const nv = v - dt * 1.25;
							if (nv <= 0) {
								flash.delete(i);
								const c = baseColor(i);
								col.set([c.r, c.g, c.b], 3 * i);
							} else {
								flash.set(i, nv);
								const c = baseColor(i);
								const k = 0.3 + 0.7 * nv;
								col.set(
									[
										c.r + (col[3 * i] - c.r) * k,
										c.g + (col[3 * i + 1] - c.g) * k,
										c.b + (col[3 * i + 2] - c.b) * k,
									],
									3 * i,
								);
							}
						}
						geo.attributes.color.needsUpdate = true;
					}
					renderer.render(scene, camera);
				};
				raf = requestAnimationFrame(tick);
			});

		const onResize = () => {
			renderer.setSize(el.clientWidth, el.clientHeight);
			frameCloud();
		};
		window.addEventListener("resize", onResize);
		return () => {
			disposed = true;
			cancelAnimationFrame(raf);
			window.removeEventListener("resize", onResize);
			renderer.dispose();
			el.removeChild(renderer.domElement);
		};
	}, []);

	useEffect(() => {
		const a = api.current;
		if (!a || activeKc.length === 0) return;
		const kcs = a.byClass.Kenyon_Cell ?? [];
		a.light(a.byClass.olfactory ?? [], CLASS_COLOR.olfactory, 0.8);
		const t1 = setTimeout(
			() => a.light(a.byClass.ALPN ?? [], CLASS_COLOR.ALPN, 0.9),
			90,
		);
		const t2 = setTimeout(
			() =>
				a.light(
					activeKc.map((i) => kcs[i % kcs.length]),
					CLASS_COLOR.Kenyon_Cell,
					1.2,
				),
			180,
		);
		return () => {
			clearTimeout(t1);
			clearTimeout(t2);
		};
	}, [activeKc]);

	useEffect(() => {
		const a = api.current;
		if (!a || !pulse) return;
		const t = setTimeout(
			() => a.light(a.byClass[pulse.cls] ?? [], pulse.color, 1.2),
			270,
		);
		return () => clearTimeout(t);
	}, [pulse]);

	return <div ref={host} className="h-full w-full" />;
}
