import { expect, test } from 'claude-code/testing'

import { difficultyMeter, holdMeter } from './lib/meters'
import { plain } from './kit'

const riskMeter = (value: number, cells?: number) => holdMeter(value, 0.9, cells)
const TIERS = { easyBelow: 0.5, hardFrom: 1.5 }

// A lit cell is a half block and an unlit one a low bar, so the level reads without colour too.
const lit = (count: number) => '▐'.repeat(count)
const unlit = (count: number) => '▁'.repeat(count)

test('the risk meter lights 16 cells with the hold line at 0.9, before the last two', () => {
  expect(plain(riskMeter(0))).toBe(`${unlit(14)}┃${unlit(2)}`)
  expect(riskMeter(0)).toEqual([
    { text: unlit(14), tone: 'rest' },
    { text: '┃', tone: 'hold' },
    { text: unlit(2), tone: 'rest' },
  ])
  expect(riskMeter(0.5)).toEqual([
    { text: lit(8), tone: 'fill' },
    { text: unlit(6), tone: 'rest' },
    { text: '┃', tone: 'hold' },
    { text: unlit(2), tone: 'rest' },
  ])
  expect(riskMeter(0.96)).toEqual([
    { text: lit(14), tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: lit(1), tone: 'fill' },
    { text: unlit(1), tone: 'rest' },
  ])
  expect(riskMeter(1)).toEqual([
    { text: lit(14), tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: lit(2), tone: 'fill' },
  ])
})

test('the risk meter lights whole cells, to the nearest, and keeps to its width', () => {
  expect(riskMeter(0.02)).toEqual(riskMeter(0))
  expect(riskMeter(0.04)[0]).toEqual({ text: lit(1), tone: 'fill' })
  expect(plain(riskMeter(0.76, 12))).toBe(`${lit(9)}${unlit(2)}┃${unlit(1)}`)
  expect(riskMeter(0.76, 12)).toEqual([
    { text: lit(9), tone: 'fill' },
    { text: unlit(2), tone: 'rest' },
    { text: '┃', tone: 'hold' },
    { text: unlit(1), tone: 'rest' },
  ])
  expect(riskMeter(-3)).toEqual(riskMeter(0))
  expect(riskMeter(9)).toEqual(riskMeter(1))
  expect(riskMeter(Number.NaN)).toEqual(riskMeter(0))
})

test('the difficulty meter lights 16 cells across the rules at 0.5 and 1.5', () => {
  expect([0.09, 0.74, 1.61].map(value => plain(difficultyMeter(value, TIERS)))).toEqual([
    `${lit(1)}${unlit(3)}│${unlit(8)}│${unlit(4)}`,
    `${lit(4)}│${lit(2)}${unlit(6)}│${unlit(4)}`,
    `${lit(4)}│${lit(8)}│${lit(1)}${unlit(3)}`,
  ])
  expect(difficultyMeter(0.09, TIERS)).toEqual([
    { text: lit(1), tone: 'fill' },
    { text: unlit(3), tone: 'rest' },
    { text: '│', tone: 'rule' },
    { text: unlit(8), tone: 'rest' },
    { text: '│', tone: 'rule' },
    { text: unlit(4), tone: 'rest' },
  ])
  expect(difficultyMeter(0.74, TIERS)).toEqual([
    { text: lit(4), tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: lit(2), tone: 'fill' },
    { text: unlit(6), tone: 'rest' },
    { text: '│', tone: 'rule' },
    { text: unlit(4), tone: 'rest' },
  ])
  expect(difficultyMeter(1.61, TIERS)).toEqual([
    { text: lit(4), tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: lit(8), tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: lit(1), tone: 'fill' },
    { text: unlit(3), tone: 'rest' },
  ])
})

test('the hold line and the tier rules move with the thresholds', () => {
  expect(holdMeter(0.85, 0.8)).toEqual([
    { text: lit(13), tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: lit(1), tone: 'fill' },
    { text: unlit(2), tone: 'rest' },
  ])
  expect(plain(difficultyMeter(0.09, { easyBelow: 1, hardFrom: 1.75 }))).toBe(`${lit(1)}${unlit(7)}│${unlit(6)}│${unlit(2)}`)
  // A line at the very end goes after the last cell.
  expect(holdMeter(0.5, 1)).toEqual([
    { text: lit(8), tone: 'fill' },
    { text: unlit(8), tone: 'rest' },
    { text: '┃', tone: 'hold' },
  ])
})

test('a score lights past its line exactly when it reaches it, however the cells round', () => {
  // 0.47 and 0.53 round to the same eight cells; the line at 0.50 keeps them apart
  expect(plain(holdMeter(0.47, 0.5))).toBe(`${lit(8)}┃${unlit(8)}`)
  expect(plain(holdMeter(0.53, 0.5))).toBe(`${lit(8)}┃${lit(1)}${unlit(7)}`)
  expect(plain(riskMeter(0.85))).toBe(`${lit(14)}┃${unlit(2)}`)
  expect(plain(riskMeter(0.9))).toBe(`${lit(14)}┃${lit(1)}${unlit(1)}`)
  // easy below 0.5, multi-step from it
  expect(plain(difficultyMeter(0.45, TIERS))).toBe(`${lit(4)}│${unlit(8)}│${unlit(4)}`)
  expect(plain(difficultyMeter(0.5, TIERS))).toBe(`${lit(4)}│${lit(1)}${unlit(7)}│${unlit(4)}`)
})
