import { useEffect, useRef } from "react";
import * as THREE from "three";

const KEY_NAME: Record<string, string> = {
	jump: "↑ UP",
	roll: "↓ DOWN",
	left: "← LEFT",
	right: "→ RIGHT",
	none: "-",
};

/**
 * A 3D fly standing on a keyboard (three.js). The arrow key for `action` lights up and a
 * foreleg taps it; between presses the fly idles (wings shimmer, body bobs).
 */
export function FlyKeyboard({
	action,
	running,
}: {
	action: string | null;
	running: boolean;
}) {
	const host = useRef<HTMLDivElement>(null);
	const api = useRef<{ press: (id: string | null) => void } | null>(null);
	const active = action && action !== "none" ? action : null;

	useEffect(() => {
		const el = host.current;
		if (!el) return;
		const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(el.clientWidth, el.clientHeight);
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFSoftShadowMap;
		el.appendChild(renderer.domElement);
		const scene = new THREE.Scene();
		const camera = new THREE.PerspectiveCamera(
			32,
			el.clientWidth / el.clientHeight,
			0.1,
			100,
		);
		camera.position.set(2.6, 3.0, 6.2);
		camera.lookAt(0.4, 0.5, 0);

		scene.add(new THREE.HemisphereLight(0xbfd8e6, 0x102018, 0.9));
		const key = new THREE.DirectionalLight(0xffffff, 1.6);
		key.position.set(3, 6, 4);
		key.castShadow = true;
		key.shadow.mapSize.set(1024, 1024);
		scene.add(key);
		const rim = new THREE.PointLight(0x4fd1c5, 4, 12);
		rim.position.set(-3, 2, -2);
		scene.add(rim);

		// desk mat
		const mat = new THREE.Mesh(
			new THREE.BoxGeometry(9, 0.12, 6),
			new THREE.MeshStandardMaterial({ color: 0x163126, roughness: 0.95 }),
		);
		mat.position.y = -0.06;
		mat.receiveShadow = true;
		scene.add(mat);

		// keyboard slab
		const slab = new THREE.Mesh(
			new THREE.BoxGeometry(4.6, 0.22, 1.9),
			new THREE.MeshStandardMaterial({
				color: 0x1a232c,
				roughness: 0.6,
				metalness: 0.25,
			}),
		);
		slab.position.set(0.3, 0.11, 0.6);
		slab.castShadow = true;
		slab.receiveShadow = true;
		scene.add(slab);

		const keyMat = () =>
			new THREE.MeshStandardMaterial({
				color: 0x2a3540,
				roughness: 0.55,
				metalness: 0.1,
			});
		const keyGeo = new THREE.BoxGeometry(0.3, 0.14, 0.3);
		for (let r = 0; r < 4; r++) {
			for (let c = 0; c < 10; c++) {
				const k = new THREE.Mesh(keyGeo, keyMat());
				k.position.set(-1.75 + c * 0.36 + r * 0.06, 0.29, 0.02 + r * 0.38);
				k.castShadow = true;
				slab.parent?.add(k);
			}
		}

		// arrow keys (inverted T) on the right of the slab
		const arrows: Record<string, THREE.Mesh> = {};
		const arrowMat = () =>
			new THREE.MeshStandardMaterial({
				color: 0x32404c,
				roughness: 0.5,
				emissive: 0x000000,
			});
		const arrowPos: Record<string, [number, number]> = {
			jump: [1.95, 0.78],
			left: [1.57, 1.16],
			roll: [1.95, 1.16],
			right: [2.33, 1.16],
		};
		for (const [id, [x, z]] of Object.entries(arrowPos)) {
			const k = new THREE.Mesh(
				new THREE.BoxGeometry(0.34, 0.14, 0.34),
				arrowMat(),
			);
			k.position.set(x, 0.29, z);
			k.castShadow = true;
			scene.add(k);
			arrows[id] = k;
		}
		// arrow glyphs as thin planes on top of the keys
		const glyph = (txt: string) => {
			const c = document.createElement("canvas");
			c.width = c.height = 64;
			const g = c.getContext("2d");
			if (g) {
				g.fillStyle = "#9fb0bb";
				g.font = "bold 44px sans-serif";
				g.textAlign = "center";
				g.textBaseline = "middle";
				g.fillText(txt, 32, 34);
			}
			const t = new THREE.CanvasTexture(c);
			return new THREE.Mesh(
				new THREE.PlaneGeometry(0.26, 0.26),
				new THREE.MeshBasicMaterial({ map: t, transparent: true }),
			);
		};
		for (const [id, sym] of Object.entries({
			jump: "▲",
			left: "◀",
			roll: "▼",
			right: "▶",
		})) {
			const p = glyph(sym);
			p.rotation.x = -Math.PI / 2;
			p.position.set(arrowPos[id][0], 0.371, arrowPos[id][1]);
			scene.add(p);
		}

		// the fly
		const fly = new THREE.Group();
		const bodyMat = new THREE.MeshStandardMaterial({
			color: 0xb9782d,
			roughness: 0.45,
			metalness: 0.15,
		});
		const thorax = new THREE.Mesh(
			new THREE.SphereGeometry(0.42, 32, 24),
			bodyMat,
		);
		thorax.scale.set(1.0, 0.85, 1.15);
		thorax.castShadow = true;
		fly.add(thorax);
		const abdomen = new THREE.Mesh(
			new THREE.SphereGeometry(0.4, 32, 24),
			new THREE.MeshStandardMaterial({ color: 0x8f5a1e, roughness: 0.5 }),
		);
		abdomen.scale.set(0.9, 0.75, 1.6);
		abdomen.position.set(0, -0.05, -0.85);
		abdomen.castShadow = true;
		fly.add(abdomen);
		// abdominal stripes
		for (let i = 0; i < 4; i++) {
			const ring = new THREE.Mesh(
				new THREE.TorusGeometry(0.33 - i * 0.04, 0.03, 8, 32),
				new THREE.MeshStandardMaterial({ color: 0x3a2308 }),
			);
			ring.position.set(0, -0.05, -0.75 - i * 0.22);
			ring.scale.set(1, 0.75, 1);
			fly.add(ring);
		}
		const head = new THREE.Mesh(
			new THREE.SphereGeometry(0.28, 32, 24),
			bodyMat,
		);
		head.position.set(0, 0.08, 0.55);
		head.castShadow = true;
		fly.add(head);
		const eyeMat = new THREE.MeshStandardMaterial({
			color: 0xd7332a,
			roughness: 0.25,
			emissive: 0x5a0d08,
		});
		for (const s of [-1, 1]) {
			const eye = new THREE.Mesh(
				new THREE.SphereGeometry(0.17, 24, 16),
				eyeMat,
			);
			eye.position.set(s * 0.2, 0.12, 0.66);
			fly.add(eye);
		}
		// wings
		const wingMat = new THREE.MeshPhysicalMaterial({
			color: 0xcfe6ff,
			transparent: true,
			opacity: 0.32,
			roughness: 0.1,
			transmission: 0.6,
			side: THREE.DoubleSide,
		});
		const wings: THREE.Mesh[] = [];
		for (const s of [-1, 1]) {
			const w = new THREE.Mesh(new THREE.CircleGeometry(0.62, 32), wingMat);
			w.scale.set(1.9, 0.75, 1);
			w.position.set(s * 0.25, 0.32, -0.25);
			w.rotation.x = -Math.PI / 2;
			w.rotation.z = s * 0.55;
			wings.push(w);
			fly.add(w);
		}
		// legs: three pairs; the front-right leg is the one that taps keys
		const legMat = new THREE.MeshStandardMaterial({ color: 0x4a2c0c });
		const legs: { pivot: THREE.Group; rest: THREE.Euler }[] = [];
		const makeLeg = (x: number, z: number, side: number, back = false) => {
			const pivot = new THREE.Group();
			pivot.position.set(x, -0.05, z);
			const femur = new THREE.Mesh(
				new THREE.CylinderGeometry(0.035, 0.03, 0.62, 8),
				legMat,
			);
			femur.position.set(side * 0.3, -0.12, 0);
			femur.rotation.z = -side * 1.15;
			const tibia = new THREE.Mesh(
				new THREE.CylinderGeometry(0.03, 0.02, 0.6, 8),
				legMat,
			);
			tibia.position.set(side * 0.6, -0.45, 0);
			tibia.rotation.z = -side * 0.35;
			pivot.add(femur, tibia);
			pivot.rotation.y = back ? -side * 0.7 : side * 0.5;
			fly.add(pivot);
			legs.push({ pivot, rest: pivot.rotation.clone() });
			return pivot;
		};
		makeLeg(0.28, 0.35, 1);
		makeLeg(-0.28, 0.35, -1);
		makeLeg(0.36, -0.05, 1);
		makeLeg(-0.36, -0.05, -1);
		makeLeg(0.3, -0.45, 1, true);
		makeLeg(-0.3, -0.45, -1, true);
		const tapLeg = legs[0].pivot;
		fly.position.set(0.55, 1.15, 0.05);
		fly.rotation.y = -0.35;
		scene.add(fly);

		let pressed: string | null = null;
		let pressT = 0;
		api.current = {
			press: (id) => {
				if (pressed && pressed !== id) {
					const m = arrows[pressed].material as THREE.MeshStandardMaterial;
					m.emissive.setHex(0x000000);
					arrows[pressed].position.y = 0.29;
				}
				pressed = id;
				pressT = performance.now();
				if (id && arrows[id]) {
					const m = arrows[id].material as THREE.MeshStandardMaterial;
					m.emissive.setHex(0x2fb8ad);
					arrows[id].position.y = 0.24;
				}
			},
		};

		let disposed = false;
		let raf = 0;
		const tick = (now: number) => {
			if (disposed) return;
			raf = requestAnimationFrame(tick);
			const t = now / 1000;
			fly.position.y = 1.15 + Math.sin(t * 2.2) * 0.03;
			for (const [i, w] of wings.entries()) {
				w.rotation.z = (i === 0 ? -1 : 1) * (0.55 + Math.sin(t * 40) * 0.08);
			}
			// tap animation: foreleg swings toward the pressed key for 250 ms
			const since = (now - pressT) / 1000;
			if (pressed && since < 0.25) {
				const k = Math.sin((since / 0.25) * Math.PI);
				const target = arrowPos[pressed];
				const dx = target[0] - (fly.position.x + 0.28);
				tapLeg.rotation.y = legs[0].rest.y + Math.atan2(dx, 1.2) * k;
				tapLeg.rotation.x = legs[0].rest.x + 0.9 * k;
			} else {
				tapLeg.rotation.copy(legs[0].rest);
				if (pressed && since > 0.6) {
					const m = arrows[pressed].material as THREE.MeshStandardMaterial;
					m.emissive.setHex(0x000000);
					arrows[pressed].position.y = 0.29;
					pressed = null;
				}
			}
			renderer.render(scene, camera);
		};
		raf = requestAnimationFrame(tick);
		const onResize = () => {
			renderer.setSize(el.clientWidth, el.clientHeight);
			camera.aspect = el.clientWidth / el.clientHeight;
			camera.updateProjectionMatrix();
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
		api.current?.press(active);
	}, [active]);

	return (
		<div className="relative h-full w-full">
			<div ref={host} className="h-full w-full" />
			<div className="absolute bottom-2 left-3 flex items-baseline gap-4 text-[11px]">
				<span className="font-bold tracking-widest text-[#4fd1c5]">
					KEY: {KEY_NAME[action ?? "none"] ?? action}
				</span>
				<span className="text-[#7a8a94]">
					{running
						? active
							? "key press registered"
							: "holding lane"
						: "idle"}
				</span>
			</div>
		</div>
	);
}
