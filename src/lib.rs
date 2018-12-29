
#[macro_use]
extern crate vst;
extern crate winapi;

use vst::api::{Events, Supported};
use vst::buffer::AudioBuffer;
use vst::channels::ChannelInfo;
use vst::editor::{Editor, KeyCode, KnobMode};
use vst::event::Event;
use vst::plugin::{CanDo, Category, HostCallback, Info, Plugin};

use winapi::um::timeapi::{timeBeginPeriod, timeEndPeriod};

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env::temp_dir;
use std::f32;
use std::fs::File;
use std::io::{stdout, Write};
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::thread;
use std::time::Duration;

#[derive(Default)]
struct Note {
	tone: u8,
	since_on: i32,
	since_off: Option<i32>,
}

#[derive(Default)]
struct ConcurrencyPlugin {
	has_editor: bool,

	current: Mutex<BTreeMap<&'static str, u8>>,
	memory: Mutex<BTreeSet<String>>,
	threads: Mutex<HashMap<thread::ThreadId, BTreeSet<&'static str>>>,

	preset_num: AtomicUsize,
	sample_rate: RwLock<f32>,
	notes: Mutex<Vec<Note>>,
	editor_open: AtomicBool,
	suspended: AtomicBool,

	output: Mutex<Option<File>>,
}

impl ConcurrencyPlugin {
	fn out(&self, s: String) {
		if let Some(file) = self.output.lock().unwrap().deref_mut() {
			file.write(s.as_bytes()).ok();
			file.write(&['\n' as u8]).ok();
			file.flush().ok();
			stdout().write(s.as_bytes()).ok();
			stdout().write(&['\n' as u8]).ok();
			stdout().flush().ok();
		}
	}

	fn up_name(&self, name: &'static str) -> Vec<&'static str> {
		// Insert function name into multiset of current functions.
		let mut current_lock = self.current.lock().unwrap();
		let current = current_lock.deref_mut();
		*current.entry(name).or_insert(0) += 1;
		let mut list = vec![];
		for (&s, &count) in current.iter() {
			for _ in 0..count {
				list.push(s);
			}
		}
		list
	}

	fn down_name(&self, name: &'static str) {
		// Remove function name from multiset of current functions.
		*self.current.lock().unwrap().deref_mut().get_mut(name).unwrap() -= 1;
	}

	fn fun<T, F: FnOnce() -> T>(&self, name: &'static str, inner: F) -> T {
		// Build string of all current functions.
		let mut set = String::new();
		for s in self.up_name(name) {
			if !set.is_empty() {
				set += " ";
			}
			set += s;
		}

		// Print this combination if it was not seen before.
		{
			let mut memory_lock = self.memory.lock().unwrap();
			let memory = memory_lock.deref_mut();
			if !memory.contains(&set) {
				self.out(format!(" *** \x1b[33m{}\x1b[0m", set));
				memory.insert(set);
			}
		}

		// Register this function for the current thread
		{
			let mut threads_lock = self.threads.lock().unwrap();
			let threads = threads_lock.deref_mut();
			let thread_id = thread::current().id();
			let thread_names = threads.entry(thread_id).or_insert(BTreeSet::new());
			if !thread_names.contains(name) {
				thread_names.insert(name);
				self.out(format!("\x1b[31m{:?}\x1b[0m: \x1b[33m{}\x1b[0m", thread::current().id(), name));
			}
		}

		// Do the actual work of the function.
		let result = inner();

		// Spend a whole millisecond to make collisions more likely.
		thread::sleep(Duration::from_millis(1));

		self.down_name(name);

		result
	}
}

impl Plugin for ConcurrencyPlugin {
	fn get_info(&self) -> Info {
		self.fun("get_info", || Info {
			name: "Concurrency".to_string(),
			vendor: "Loonies".to_string(),
			presets: 2,
			parameters: 3,
			inputs: 0,
			outputs: 2,
			unique_id: 0xC0C0,
			version: 1000,
			category: Category::Synth,
			preset_chunks: true,

			.. Info::default()
		})
	}

	fn new(_host: HostCallback) -> Self {
		let mut plugin = ConcurrencyPlugin::default();
		plugin.has_editor = true;
		if !plugin.suspended.swap(true, Ordering::Relaxed) {
			plugin.up_name("(suspended)");
		}
		plugin
	}

	fn init(&mut self) {
		self.fun("init", || {
			let mut path = temp_dir();
			path.push("vst_concurrency.txt");
			*self.output.lock().unwrap().deref_mut() = File::create(path).ok();
		});
	}

	fn change_preset(&mut self, preset: i32) {
		self.fun("change_preset", || {
			self.preset_num.store(preset as usize, Ordering::Relaxed)
		})
	}

	fn get_preset_num(&self) -> i32 {
		self.fun("get_preset_num", || {
			self.preset_num.load(Ordering::Relaxed) as i32
		})
	}

	fn set_preset_name(&mut self, _name: String) {
		self.fun("set_preset_name", || ())
	}

	fn get_preset_name(&self, preset: i32) -> String {
		self.fun("get_preset_name", || format!("preset_{}", preset))
	}

	fn get_parameter_label(&self, index: i32) -> String {
		self.fun("get_parameter_label", || format!("pl_{}", index))
	}

	fn get_parameter_text(&self, index: i32) -> String {
		self.fun("get_parameter_text", || format!("pt_{}", index))
	}

	fn get_parameter_name(&self, index: i32) -> String {
		self.fun("get_parameter_name", || format!("pn_{}", index))
	}

	fn get_parameter(&self, _index: i32) -> f32 {
		self.fun("get_parameter", || 0.0)
	}

	fn set_parameter(&mut self, _index: i32, _value: f32) {
		self.fun("set_parameter", || ())
	}

	fn can_be_automated(&self, _index: i32) -> bool {
		self.fun("can_be_automated", || true)
	}

	fn string_to_parameter(&mut self, _index: i32, _text: String) -> bool {
		self.fun("string_to_parameter", || true)
	}

	fn set_sample_rate(&mut self, rate: f32) {
		self.fun("set_sample_rate", || {
			*self.sample_rate.write().unwrap().deref_mut() = rate;
		})
	}

	fn set_block_size(&mut self, _size: i64) {
		self.fun("set_block_size", || ())
	}

	fn resume(&mut self) {
		#[cfg(windows)] unsafe { timeBeginPeriod(1) };
		self.out(format!("\x1b[32mResumed!\x1b[0m"));
		self.fun("resume", || {
			if self.suspended.swap(false, Ordering::Relaxed) {
				self.down_name("(suspended)");
			}
		})
	}

	fn suspend(&mut self) {
		#[cfg(windows)] unsafe { timeEndPeriod(1) };
		self.out(format!("\x1b[32mSuspended!\x1b[0m"));
		self.fun("suspend", || {
			if !self.suspended.swap(true, Ordering::Relaxed) {
				self.up_name("(suspended)");
			}
		})
	}

	fn can_do(&self, can_do: CanDo) -> Supported {
		self.fun("can_do", || {
			match can_do {
				CanDo::ReceiveEvents => Supported::Yes,
				CanDo::ReceiveMidiEvent => Supported::Yes,
				_ => Supported::Maybe
			}
		})
	}

	fn get_tail_size(&self) -> isize {
		self.fun("get_tail_size", || 0)
	}

	fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
		self.fun("process", || {
			let mut notes_lock = self.notes.lock().unwrap();
			let notes = notes_lock.deref_mut();
			let samples = buffer.samples() as i32;
			let sample_rate = *self.sample_rate.read().unwrap().deref();
			let (_inputs, outputs) = buffer.split();
			for i in 0..samples {
				let mut sum = 0f32;
				for n in notes.iter() {
					let n_on = i + n.since_on;
					if n_on >= 0 {
						// Simple sine wave with one-second release.
						let freq = 2f32.powf((n.tone as i32 - 69) as f32 / 12f32) * 440f32;
						let phase = n_on as f32 / sample_rate * freq * 2f32 * f32::consts::PI;
						let mut s = phase.sin();
						if let Some(since_off) = n.since_off {
							let n_off = i + since_off;
							s *= (1f32 - (n_off as f32 / sample_rate)).max(0f32).min(1f32);
						}
						sum += s;
					}
				}
				for o in 0..outputs.len() {
					outputs.get_mut(o)[i as usize] = sum;
				}
			}
			for n in notes.iter_mut() {
				n.since_on += samples;
				n.since_off = n.since_off.map(|since_off| since_off + samples);
			}
			notes.retain(|n| n.since_off.map_or(true, |since_off| since_off < sample_rate as i32));
		})
	}

	fn process_f64(&mut self, _buffer: &mut AudioBuffer<f64>) {
		panic!("64-bit processing not supported");
	}

	fn process_events(&mut self, events: &Events) {
		self.fun("process_events", || {
			let mut notes_lock = self.notes.lock().unwrap();
			let notes = notes_lock.deref_mut();
			for event in events.events() {
				if let Event::Midi(midi) = event {
					match midi.data[0] & 0xF0 {
						// Note On
						0x90 => notes.push(Note {
							tone: midi.data[1],
							since_on: -midi.delta_frames,
							since_off: None,
						}),
						// Note Off
						0x80 => for n in notes.iter_mut() {
							if n.tone == midi.data[1] {
								n.since_off = Some(-midi.delta_frames);
							}
						},
						_ => {
							self.out(format!("Midi event: {:02X} {:02X} {:02X}", midi.data[0], midi.data[1], midi.data[2]));
						}
					}
				}
			}
		})
	}

	fn get_editor(&mut self) -> Option<&mut Editor> {
		if self.has_editor {
			Some(self)
		} else {
			None
		}
	}

	fn get_preset_data(&mut self) -> Vec<u8> {
		self.fun("get_preset_data", || vec![1, 2, 3])
	}

	fn get_bank_data(&mut self) -> Vec<u8> {
		self.fun("get_bank_data", || vec![1, 2, 3])
	}

	fn load_preset_data(&mut self, _data: &[u8]) {
		self.fun("load_preset_data", || ())
	}

	fn load_bank_data(&mut self, _data: &[u8]) {
		self.fun("load_bank_data", || ())
	}

	fn get_input_info(&self, _input: i32) -> ChannelInfo {
		self.fun("get_input_info", || {
			ChannelInfo::new("Foo".to_string(), None, true, None)
		})
	}

	fn get_output_info(&self, _output: i32) -> ChannelInfo {
		self.fun("get_output_info", || {
			ChannelInfo::new("Foo".to_string(), None, true, None)
		})
	}

	fn start_process(&self) {
		self.fun("start_process", || ())
	}

	fn stop_process(&self) {
		self.fun("stop_process", || ())
	}
}

impl Editor for ConcurrencyPlugin {
	fn size(&self) -> (i32, i32) {
		self.fun("size", || (400, 100))
	}

	fn position(&self) -> (i32, i32) {
		self.fun("position", || (0, 0))
	}

	fn open(&mut self, _window: *mut c_void) {
		self.fun("open", || self.editor_open.store(true, Ordering::Relaxed))
	}

	fn is_open(&mut self) -> bool {
		self.fun("is_open", || self.editor_open.load(Ordering::Relaxed))
	}

	fn idle(&mut self) {
		self.fun("idle", || ())
	}

	fn close(&mut self) {
		self.fun("close", || self.editor_open.store(false, Ordering::Relaxed))
	}

	fn set_knob_mode(&mut self, _mode: KnobMode) -> bool {
		self.fun("set_knob_mode", || true)
	}

	fn key_up(&mut self, _keycode: KeyCode) -> bool {
		self.fun("key_up", || false)
	}

	fn key_down(&mut self, _keycode: KeyCode) -> bool {
		self.fun("key_down", || false)
	}
}

plugin_main!(ConcurrencyPlugin);
