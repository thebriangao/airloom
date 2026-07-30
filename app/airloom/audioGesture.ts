export type SoundSample = {
  transient: boolean;
  peak: number;
  rms: number;
};

export class SuddenSoundDetector {
  private baseline = 0.006;
  private previousPeak = 0;
  private previousRms = 0;
  private armed = true;
  private lastTransientAt = -Infinity;

  update(samples: Float32Array, timestamp: number): SoundSample {
    let peak = 0;
    let squareSum = 0;
    for (const sample of samples) {
      const magnitude = Math.abs(sample);
      peak = Math.max(peak, magnitude);
      squareSum += sample * sample;
    }

    const rms = Math.sqrt(squareSum / Math.max(1, samples.length));
    const baseline = this.baseline;
    const quiet =
      rms < Math.max(0.012, baseline * 1.6) &&
      peak < Math.max(0.04, baseline * 3);

    if (quiet) {
      this.baseline = baseline * 0.96 + rms * 0.04;
      this.armed = true;
    }

    const loudEnough =
      peak > Math.max(0.05, baseline * 3.3) &&
      rms > Math.max(0.008, baseline * 1.65);
    const suddenRise =
      peak > Math.max(this.previousPeak * 1.55, baseline * 3.3) ||
      rms > Math.max(this.previousRms * 1.8, baseline * 1.65);
    const transient =
      this.armed &&
      loudEnough &&
      suddenRise &&
      timestamp - this.lastTransientAt > 160;

    if (transient) {
      this.armed = false;
      this.lastTransientAt = timestamp;
    }

    this.previousPeak = peak;
    this.previousRms = rms;
    return { transient, peak, rms };
  }

  reset() {
    this.baseline = 0.006;
    this.previousPeak = 0;
    this.previousRms = 0;
    this.armed = true;
    this.lastTransientAt = -Infinity;
  }
}
