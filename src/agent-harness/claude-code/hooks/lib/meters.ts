import type { Tiers } from '../../types'
import { DIFFICULTY_MAX } from './settings'

export type Tone = 'fill' | 'rest' | 'rule' | 'hold'
export type Segment = { text: string; tone: Tone }

const CELLS = 16
// A right half block, so lit and unlit cells read as separate pixels.
const CELL = '▐'

// One cell for each `max / cells`, lit up to `value`. `cuts` put a glyph before the cell at each `at`, or after the last
// cell for a line at the very end.
function meterOf(value: number, max: number, cells: number, cuts: { at: number; glyph: string; tone: Tone }[]): Segment[] {
  const ratio = Number.isFinite(value) ? Math.min(1, Math.max(0, value / max)) : 0
  const lit = Math.round(ratio * cells)
  const segments: Segment[] = []
  const push = (text: string, tone: Tone) => {
    const last = segments.at(-1)
    if (last?.tone === tone) last.text += text
    else segments.push({ text, tone })
  }
  for (let i = 0; i <= cells; i++) {
    for (const cut of cuts) if (cut.at === i) push(cut.glyph, cut.tone)
    if (i < cells) push(CELL, i < lit ? 'fill' : 'rest')
  }
  return segments
}

// Difficulty, 0 to 2: the rules mark the tier lines.
export const difficultyMeter = (value: number, { easyBelow, hardFrom }: Tiers) =>
  meterOf(value, DIFFICULTY_MAX, CELLS, [easyBelow, hardFrom].map(line => ({
    at: Math.round((line / DIFFICULTY_MAX) * CELLS),
    glyph: '│',
    tone: 'rule' as const,
  })))

// A score from 0 to 1 with its hold line where `line` falls.
export const holdMeter = (value: number, line: number, cells = CELLS) =>
  meterOf(value, 1, cells, [{ at: Math.round(line * cells), glyph: '┃', tone: 'hold' }])

export const textWidth = (segments: Segment[]) => segments.reduce((sum, one) => sum + one.text.length, 0)
