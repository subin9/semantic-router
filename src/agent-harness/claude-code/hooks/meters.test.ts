import { expect, test } from 'claude-code/testing'

import { difficultyMeter, holdMeter } from './lib/meters'
import { plain } from './kit'

const riskMeter = (value: number, cells?: number) => holdMeter(value, 0.9, cells)
const TIERS = { easyBelow: 0.5, hardFrom: 1.5 }

// Every cell is the same half block, lit or not; the tone tells them apart.
const cells = (count: number) => '▐'.repeat(count)

test('the risk meter lights 16 cells with the hold line at 0.9, before the last two', () => {
  expect(plain(riskMeter(0))).toBe(`${cells(14)}┃${cells(2)}`)
  expect(riskMeter(0)).toEqual([
    { text: cells(14), tone: 'rest' },
    { text: '┃', tone: 'hold' },
    { text: cells(2), tone: 'rest' },
  ])
  expect(riskMeter(0.5)).toEqual([
    { text: cells(8), tone: 'fill' },
    { text: cells(6), tone: 'rest' },
    { text: '┃', tone: 'hold' },
    { text: cells(2), tone: 'rest' },
  ])
  expect(riskMeter(0.96)).toEqual([
    { text: cells(14), tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: cells(1), tone: 'fill' },
    { text: cells(1), tone: 'rest' },
  ])
  expect(riskMeter(1)).toEqual([
    { text: cells(14), tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: cells(2), tone: 'fill' },
  ])
})

test('the risk meter lights whole cells, to the nearest, and keeps to its width', () => {
  expect(riskMeter(0.02)).toEqual(riskMeter(0))
  expect(riskMeter(0.04)[0]).toEqual({ text: cells(1), tone: 'fill' })
  expect(plain(riskMeter(0.76, 12))).toBe(`${cells(11)}┃${cells(1)}`)
  expect(riskMeter(0.76, 12)).toEqual([
    { text: cells(9), tone: 'fill' },
    { text: cells(2), tone: 'rest' },
    { text: '┃', tone: 'hold' },
    { text: cells(1), tone: 'rest' },
  ])
  expect(riskMeter(-3)).toEqual(riskMeter(0))
  expect(riskMeter(9)).toEqual(riskMeter(1))
  expect(riskMeter(Number.NaN)).toEqual(riskMeter(0))
})

test('the difficulty meter lights 16 cells across the rules at 0.5 and 1.5', () => {
  for (const value of [0.09, 0.74, 1.61]) expect(plain(difficultyMeter(value, TIERS))).toBe(`${cells(4)}│${cells(8)}│${cells(4)}`)
  expect(difficultyMeter(0.09, TIERS)).toEqual([
    { text: cells(1), tone: 'fill' },
    { text: cells(3), tone: 'rest' },
    { text: '│', tone: 'rule' },
    { text: cells(8), tone: 'rest' },
    { text: '│', tone: 'rule' },
    { text: cells(4), tone: 'rest' },
  ])
  expect(difficultyMeter(0.74, TIERS)).toEqual([
    { text: cells(4), tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: cells(2), tone: 'fill' },
    { text: cells(6), tone: 'rest' },
    { text: '│', tone: 'rule' },
    { text: cells(4), tone: 'rest' },
  ])
  expect(difficultyMeter(1.61, TIERS)).toEqual([
    { text: cells(4), tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: cells(8), tone: 'fill' },
    { text: '│', tone: 'rule' },
    { text: cells(1), tone: 'fill' },
    { text: cells(3), tone: 'rest' },
  ])
})

test('the hold line and the tier rules move with the thresholds', () => {
  expect(holdMeter(0.85, 0.8)).toEqual([
    { text: cells(13), tone: 'fill' },
    { text: '┃', tone: 'hold' },
    { text: cells(1), tone: 'fill' },
    { text: cells(2), tone: 'rest' },
  ])
  expect(plain(difficultyMeter(0.09, { easyBelow: 1, hardFrom: 1.75 }))).toBe(`${cells(8)}│${cells(6)}│${cells(2)}`)
  // A line at the very end goes after the last cell.
  expect(holdMeter(0.5, 1)).toEqual([
    { text: cells(8), tone: 'fill' },
    { text: cells(8), tone: 'rest' },
    { text: '┃', tone: 'hold' },
  ])
})
