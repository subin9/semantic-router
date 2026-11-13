import type { Tiers } from '../../types'
import { DIFFICULTY_MAX } from './settings'

export type Tone = 'fill' | 'rest' | 'rule' | 'hold'
export type Segment = { text: string; tone: Tone }

const CELLS = 16
// A right half block lit, a low eighth block unlit: the level reads without colour too (copied text, no colour).
const CELL = '▐'
const UNLIT = '▁'

type Cut = { line: number; glyph: string; tone: Tone }

// The cell a line is drawn before.
const cellOf = (line: number, max: number, cells: number) => Math.round((line / max) * cells)

// One cell for each `max / cells`, lit up to `value`. Each cut puts a glyph before the cell where its line falls, or
// after the last cell for a line at the very end, and a value lights past a line exactly when it reaches it, whatever
// the rounding: 0.47 and 0.53 either side of 0.50 never draw the same cells.
function meterOf(value: number, max: number, cells: number, cuts: Cut[]): Segment[] {
  const ratio = Number.isFinite(value) ? Math.min(1, Math.max(0, value / max)) : 0
  let lit = Math.round(ratio * cells)
  for (const { line } of cuts) {
    const at = cellOf(line, max, cells)
    lit = ratio * max >= line ? Math.max(lit, Math.min(cells, at + 1)) : Math.min(lit, at)
  }
  const segments: Segment[] = []
  const push = (text: string, tone: Tone) => {
    const last = segments.at(-1)
    if (last?.tone === tone) last.text += text
    else segments.push({ text, tone })
  }
  for (let i = 0; i <= cells; i++) {
    for (const cut of cuts) if (cellOf(cut.line, max, cells) === i) push(cut.glyph, cut.tone)
    if (i < cells) push(i < lit ? CELL : UNLIT, i < lit ? 'fill' : 'rest')
  }
  return segments
}

// Difficulty, 0 to 2: the rules mark the tier lines.
export const difficultyMeter = (value: number, { easyBelow, hardFrom }: Tiers) =>
  meterOf(value, DIFFICULTY_MAX, CELLS, [easyBelow, hardFrom].map(line => ({ line, glyph: '│', tone: 'rule' as const })))

// A score from 0 to 1 with its hold line where `line` falls.
export const holdMeter = (value: number, line: number, cells = CELLS) =>
  meterOf(value, 1, cells, [{ line, glyph: '┃', tone: 'hold' }])

export const textWidth = (segments: Segment[]) => segments.reduce((sum, one) => sum + one.text.length, 0)
