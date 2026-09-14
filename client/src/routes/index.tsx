import { createFileRoute } from "@tanstack/react-router";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useFlyWs } from "#/lib/flyws";

export const Route = createFileRoute("/")({ component: Dashboard });

// both panels use WebGL, so they load on the client only
const SomaCloud = lazy(() =>
	import("#/components/SomaCloud").then((m) => ({ default: m.SomaCloud })),
);
const FlyKeyboard = lazy(() =>
	import("#/components/FlyKeyboard").then((m) => ({ default: m.FlyKeyboard })),
);

const RUN_SECONDS = 30;

function Dashboard() {
	const { connected, status, frame, step, stepCount, episodes, send } =
		useFlyWs();
	const [pulse, setPulse] = useState<{
		cls: string;
		color: number;
		key: number;
	} | null>(null);
	const [mounted, setMounted] = useState(false);
	useEffect(() => setMounted(true), []);

	// MBON pulse on every decision, DAN pulse when an episode ends (dopamine = reinforcement)
	useEffect(() => {
		if (step)
			setPulse({
				cls: "MBON",
				color: step.action === "none" ? 0x7a8a94 : 0xf687b3,
				key: stepCount,
			});
	}, [stepCount, step]);
	useEffect(() => {
		if (episodes.length)
			setPulse({ cls: "DAN", color: 0xfc8181, key: -episodes.length });
	}, [episodes.length]);

	const activeKc = useMemo(() => step?.kc_idx ?? [], [step]);
	const last = episodes[episodes.length - 1];
	const best = episodes.reduce((m, e) => Math.max(m, e.score), 0);
	const secondsLeft = status.running ? status.seconds_left : RUN_SECONDS;
	const elapsed = status.running ? RUN_SECONDS - status.seconds_left : 0;
	const phase =
		status.session === "error"
			? "BROWSER ERROR"
			: status.running
				? (status.message ?? "playing").toUpperCase()
				: status.session === "ready"
					? status.message === "parking"
						? "PARKING"
						: "READY"
					: "STARTING BROWSER";

	return (
		<div className="flex h-screen flex-col bg-[#0b0f14] font-mono text-[13px] text-[#cfd8dc]">
			<header className="flex items-center justify-between border-b border-[#1e2a33] px-5 py-3">
				<div className="flex items-baseline gap-4">
					<span className="text-2xl font-bold tracking-[0.2em] text-[#4fd1c5]">
						FLYSURF
					</span>
					<span className="text-sm tracking-[0.2em] text-[#aab7c0]">
						CONNECTOME &gt; FLY &gt; KEYBOARD &gt; SUBWAY SURFERS
					</span>
				</div>
				<div className="flex items-center gap-5">
					<span
						className={`text-[11px] tracking-widest ${connected ? "text-[#68d391]" : "text-[#fc8181]"}`}
					>
						{connected ? "● LINK" : "○ NO LINK"}
					</span>
					<span className="text-xl text-[#4fd1c5]">
						t = {elapsed.toFixed(2).padStart(5, "0")} s
					</span>
				</div>
			</header>
			<div className="border-b border-[#1e2a33] px-5 py-1 text-[11px] text-[#7a8a94]">
				Neural activity | Keyboard input | Game frames · {phase}
			</div>

			<main className="grid min-h-0 flex-1 grid-cols-[1.45fr_1fr] gap-3 p-3">
				<section className="relative flex flex-col overflow-hidden rounded border border-[#1e2a33] bg-black">
					<PaneTitle>SUBWAY SURFERS / POKI.COM · LIVE FRAMES</PaneTitle>
					<div className="relative min-h-0 flex-1">
						{frame ? (
							<img
								src={`data:image/jpeg;base64,${frame}`}
								alt="game"
								className="h-full w-full object-contain"
							/>
						) : (
							<div className="flex h-full items-center justify-center text-[#55666f]">
								waiting for the browser…
							</div>
						)}
						<Countdown secondsLeft={secondsLeft} />
					</div>
					<div className="flex items-center justify-between border-t border-[#1e2a33] bg-[#0e151b] px-4 py-2">
						<span className="text-base font-bold tracking-widest text-[#4fd1c5]">
							{phase}
						</span>
						<span className="text-[#7a8a94]">
							{status.running
								? `${step?.steps ?? 0} decisions · score ${step?.score ?? 0}`
								: last
									? `last run ${last.score} pts in ${last.wall_s.toFixed(1)} s`
									: "no run yet"}
						</span>
					</div>
				</section>

				<div className="grid min-h-0 grid-rows-[1.2fr_1fr] gap-3">
					<section className="relative flex flex-col overflow-hidden rounded border border-[#1e2a33] bg-[#070b0f]">
						<PaneTitle>CNS / NEURAL ACTIVITY</PaneTitle>
						<div className="min-h-0 flex-1">
							{mounted && (
								<Suspense fallback={null}>
									<SomaCloud activeKc={activeKc} pulse={pulse} />
								</Suspense>
							)}
						</div>
						<div className="flex items-center justify-between border-t border-[#1e2a33] px-4 py-2 text-[11px]">
							<span className="text-[#aab7c0]">
								Brain
								<span className="ml-2 text-[#7a8a94]">
									FlyWire v783 · {step ? step.kc_idx.length : 0} Kenyon cells
									active
								</span>
							</span>
							<span className="flex gap-3">
								<Legend color="#4fd1c5" label="ORN" />
								<Legend color="#63b3ed" label="PN" />
								<Legend color="#f6e05e" label="KC" />
								<Legend color="#f687b3" label="MBON" />
								<Legend color="#fc8181" label="DAN" />
							</span>
						</div>
					</section>

					<section className="relative flex flex-col overflow-hidden rounded border border-[#1e2a33] bg-[#0b1410]">
						<PaneTitle>FLY / KEYBOARD INPUT</PaneTitle>
						<div className="min-h-0 flex-1">
							{mounted && (
								<Suspense fallback={null}>
									<FlyKeyboard
										action={status.running ? (step?.action ?? null) : null}
										running={status.running}
									/>
								</Suspense>
							)}
						</div>
					</section>
				</div>
			</main>

			<section className="grid grid-cols-6 gap-3 border-t border-[#1e2a33] px-5 py-3">
				<Stat
					label="KENYON CELLS ACTIVE"
					value={step ? step.kc_idx.length : 0}
				/>
				<Stat label="SCORE" value={step?.score ?? last?.score ?? 0} />
				<Stat
					label="DECISIONS / S"
					value={last ? last.decisions_per_s.toFixed(1) : "—"}
				/>
				<Stat
					label="|RPE| LOSS"
					value={
						step ? step.loss.toFixed(3) : last ? last.loss.toFixed(3) : "—"
					}
				/>
				<Stat label="BEST SCORE" value={best} />
				<div className="flex items-center justify-end">
					<button
						type="button"
						disabled={
							!connected || status.running || status.session !== "ready"
						}
						onClick={() =>
							send({ cmd: "run", seconds: RUN_SECONDS, learning: true })
						}
						className="rounded border border-[#4fd1c5] px-5 py-2 text-sm font-bold tracking-widest text-[#4fd1c5] hover:bg-[#4fd1c5]/10 disabled:cursor-not-allowed disabled:opacity-40"
					>
						{status.running
							? `PLAYING ${Math.ceil(secondsLeft)}s`
							: `RUN ${RUN_SECONDS}s`}
					</button>
					{status.running && (
						<button
							type="button"
							onClick={() => send({ cmd: "stop" })}
							className="ml-2 rounded border border-[#fc8181] px-3 py-2 text-[11px] tracking-widest text-[#fc8181]"
						>
							STOP
						</button>
					)}
				</div>
			</section>

			<section className="max-h-24 overflow-y-auto border-t border-[#1e2a33] px-5 py-2 text-[11px]">
				<div className="mb-1 tracking-widest text-[#7a8a94]">
					EPISODES (learning curve, newest last)
				</div>
				{episodes.length === 0 ? (
					<span className="text-[#55666f]">none yet</span>
				) : (
					<div className="flex flex-wrap gap-2">
						{episodes.slice(-16).map((e) => (
							<span
								key={`${e.episode}-${e.wall_s}`}
								className="rounded border border-[#1e2a33] px-2 py-1"
							>
								<b className="text-white">{e.score}</b> pts ·{" "}
								{e.wall_s.toFixed(1)}s · {e.dead ? "crash" : "timeout"} · loss{" "}
								{e.loss.toFixed(3)}
							</span>
						))}
					</div>
				)}
			</section>

			<footer className="border-t border-[#1e2a33] px-5 py-1 text-[10px] text-[#55666f]">
				Fly at the keyboard | Subway Surfers on poki.com | Synchronized neural
				activity · FlyWire v783 (CC-BY 4.0) · ORN→PN→KC→APL→MBON rate model ·
				KC→MBON three-factor plasticity, no backprop
			</footer>
		</div>
	);
}

function PaneTitle({ children }: { children: React.ReactNode }) {
	return (
		<div className="border-b border-[#1e2a33] bg-[#0e151b] px-4 py-1.5 text-[11px] font-bold tracking-widest text-[#aab7c0]">
			{children}
		</div>
	);
}

function Legend({ color, label }: { color: string; label: string }) {
	return (
		<span className="flex items-center gap-1 text-[#7a8a94]">
			<span
				className="inline-block h-2 w-2 rounded-full"
				style={{ background: color }}
			/>
			{label}
		</span>
	);
}

function Countdown({ secondsLeft }: { secondsLeft: number }) {
	const ring = 2 * Math.PI * 26;
	return (
		<svg
			className="absolute top-3 right-3"
			width="64"
			height="64"
			viewBox="0 0 64 64"
			role="img"
			aria-label="seconds left"
		>
			<circle
				cx="32"
				cy="32"
				r="26"
				stroke="#1e2a33"
				strokeWidth="5"
				fill="none"
			/>
			<circle
				cx="32"
				cy="32"
				r="26"
				stroke="#4fd1c5"
				strokeWidth="5"
				fill="none"
				strokeDasharray={ring}
				strokeDashoffset={ring * (1 - secondsLeft / RUN_SECONDS)}
				transform="rotate(-90 32 32)"
			/>
			<text
				x="32"
				y="37"
				textAnchor="middle"
				fill="#fff"
				fontSize="16"
				fontWeight="bold"
			>
				{Math.ceil(secondsLeft)}
			</text>
		</svg>
	);
}

function Stat({ label, value }: { label: string; value: string | number }) {
	return (
		<div>
			<b className="block text-xl text-white">{value}</b>
			<span className="text-[10px] tracking-widest text-[#7a8a94]">
				{label}
			</span>
		</div>
	);
}
