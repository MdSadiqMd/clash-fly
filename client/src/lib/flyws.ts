import { useCallback, useEffect, useRef, useState } from "react";

export type Status = {
	running: boolean;
	seconds_left: number;
	session: "starting" | "ready" | "error";
	message?: string;
};

export type Step = {
	action: string;
	reward: number;
	score: number;
	elapsed: number;
	kc_idx: number[];
	scores: number[];
	loss: number;
	steps: number;
};

export type Episode = {
	episode: number;
	steps: number;
	reward: number;
	coins: number;
	score: number;
	dead: boolean;
	loss: number;
	wall_s: number;
	decisions_per_s: number;
};

type Msg =
	| ({ type: "status" } & Status)
	| { type: "frame"; jpg: string }
	| ({ type: "step" } & Step)
	| ({ type: "episode" } & Episode)
	| { type: "history"; episodes: Episode[] };

/** Live link to `flyclash surf-serve`. Reconnects on its own. */
export function useFlyWs(url = "ws://localhost:8765") {
	const [connected, setConnected] = useState(false);
	const [status, setStatus] = useState<Status>({
		running: false,
		seconds_left: 0,
		session: "starting",
	});
	const [frame, setFrame] = useState<string | null>(null);
	const [step, setStep] = useState<Step | null>(null);
	const [episodes, setEpisodes] = useState<Episode[]>([]);
	const [stepCount, setStepCount] = useState(0);
	const ws = useRef<WebSocket | null>(null);

	useEffect(() => {
		let closed = false;
		let timer: ReturnType<typeof setTimeout>;
		const connect = () => {
			const sock = new WebSocket(url);
			ws.current = sock;
			sock.onopen = () => setConnected(true);
			sock.onclose = () => {
				setConnected(false);
				if (!closed) timer = setTimeout(connect, 1500);
			};
			sock.onmessage = (ev) => {
				const m = JSON.parse(ev.data) as Msg;
				if (m.type === "status") setStatus(m);
				else if (m.type === "frame") setFrame(m.jpg);
				else if (m.type === "step") {
					setStep(m);
					setStepCount((n) => n + 1);
				} else if (m.type === "episode")
					setEpisodes((e) => [...e, m].slice(-50));
				else if (m.type === "history") setEpisodes(m.episodes);
			};
		};
		connect();
		return () => {
			closed = true;
			clearTimeout(timer);
			ws.current?.close();
		};
	}, [url]);

	const send = useCallback((cmd: Record<string, unknown>) => {
		ws.current?.send(JSON.stringify(cmd));
	}, []);

	return { connected, status, frame, step, stepCount, episodes, send };
}
