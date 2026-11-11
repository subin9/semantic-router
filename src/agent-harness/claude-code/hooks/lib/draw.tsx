import type { EngineInterface } from 'claude-code'

import type { App, Calibration, Decision, Spend } from '../../types'
import { APP_NAMES, APPS, type Color, LOOKS, type Section, SECTIONS, sectionOf } from './apps'
import { type Body, type Described, describe, type Gauge, TALLY_DOT, type TallyItem, tallyOf, tallyWidth, type Weight } from './describe'
import { ANSWERS, type Answer, answersFor } from './labels'
import { BRAND, brand, type Logo, wordmark, wordmarkHeight } from './logo'
import { holdMeter, type Segment, textWidth } from './meters'
import { choiceOf, type Display, isCleartextRemote, type Settings, WAIT_APPS, WAIT_CHOICES, type WaitApp, type WaitChoice } from './settings'
import { clip, fixed, modelName } from './text'

export type Elements = ReturnType<EngineInterface['ui']['resolve']>

const MIN_DETAIL_WIDTH = 8
// Below this the inline decision is the one-line form: a card would wrap.
const MIN_CARD_COLUMNS = 56
// Its lines under the pill keep the 72 columns they had inside the frame the card used to have.
const MAX_CARD_WIDTH = 79
// A card's lines under its header start under its pill.
const CARD_INDENT = `${BRAND} `.length
const COMPACT_RISK_CELLS = 12

// The one-line form: the mark, the outcome in its color, the headline, so it reads "vSR redacted ...". The detail gives
// way before the headline does.
function badgeLine(elements: Elements, { badge, color, headline, detail }: Described, columns: number) {
  const { Text } = elements
  const word = badge.toLowerCase()
  const room = columns - `${BRAND} ${word}  `.length
  const head = clip(headline, Math.max(1, room))
  const roomLeft = room - head.length - 2
  return (
    <Text>
      {brand(elements)}
      {' '}
      <Text color={color}>{word}</Text>
      {`  ${head}`}
      {detail !== '' && roomLeft >= MIN_DETAIL_WIDTH ? <Text dimColor>{`  ${clip(detail, roomLeft)}`}</Text> : null}
    </Text>
  )
}

// The outcome of a decision that asks for a look, as a pill in its color.
function pill({ Text }: Elements, badge: string, color: Color) {
  return <Text backgroundColor={color} color="inverseText" weight="bold">{` ${badge.toLowerCase()} `}</Text>
}

function meterCells({ Text }: Elements, segments: Segment[], fill: Color) {
  return segments.map(({ text, tone }) => (
    <Text color={tone === 'fill' ? fill : tone === 'hold' ? 'warning' : undefined} dimColor={tone === 'rest' || tone === 'rule'}>
      {text}
    </Text>
  ))
}

// The meter's own line: the score and its caption give way before the meter does.
function meterLine(elements: Elements, { lead, close, segments, fill, fits }: Gauge, room: number) {
  const { Text } = elements
  const used = lead.length + textWidth(segments) + close.length
  const [score, caption] = fits.find(([one, rest]) => used + 2 + one.length + rest.length <= room) ?? fits[fits.length - 1]
  return (
    <Text>
      <Text dimColor>{lead}</Text>
      {meterCells(elements, segments, fill)}
      <Text dimColor>{close}</Text>
      {'  '}
      <Text weight="bold">{score}</Text>
      {caption === '' ? null : <Text dimColor>{caption}</Text>}
    </Text>
  )
}

// Small badges in the card's color, as many as fit, then how many did not.
function chipsLine({ Text }: Elements, chips: string[], color: Color, room: number) {
  const shown: string[] = []
  let used = 0
  for (const [index, chip] of chips.entries()) {
    const width = chip.length + (shown.length > 0 ? ' · '.length : 0)
    const rest = chips.length - index - 1
    if (used + width + (rest > 0 ? ` +${rest}`.length : 0) > room && shown.length > 0) break
    shown.push(chip)
    used += width
  }
  const hidden = chips.length - shown.length
  return (
    <Text>
      {shown.flatMap((chip, index) => [index === 0 ? '' : <Text dimColor>{' · '}</Text>, <Text color={color}>{chip}</Text>])}
      {hidden > 0 ? <Text dimColor>{` +${hidden}`}</Text> : null}
    </Text>
  )
}

function bodyLine(elements: Elements, body: Body, color: Color, room: number) {
  return body.kind === 'gauge' ? meterLine(elements, body.gauge, room) : chipsLine(elements, body.chips, color, room)
}

// A decision worth a look: the mark and its outcome as a pill, then its meter or chips and its note, under the title.
// No frame: the pill and the indent set it apart from Claude's words in three rows.
function card(
  elements: Elements,
  { badge }: Described,
  one: Extract<Described['inline'], { kind: 'card' }>,
  columns: number,
) {
  const { Box, Text } = elements
  const width = Math.min(columns, MAX_CARD_WIDTH)
  const inner = width - CARD_INDENT
  const tail = one.tail ?? ''
  const title = `${clip(one.title, Math.max(1, inner - ` ${badge} `.length - 1 - tail.length))}${tail}`
  return (
    <Box flexDirection="column" width={width}>
      <Text>
        {brand(elements)}
        {' '}
        {pill(elements, badge, one.color)}
        {` ${title}`}
      </Text>
      <Box flexDirection="column" paddingLeft={CARD_INDENT}>
        {one.body ? bodyLine(elements, one.body, one.color, inner) : null}
        <Text dimColor>{clip(one.note, inner)}</Text>
      </Box>
    </Box>
  )
}

const COMPACT_LEAD = `${BRAND} allowed  `

function compactLine(elements: Elements, { subject, more = '' }: Described, score: number, threshold: number, columns: number) {
  const { Text } = elements
  const segments = holdMeter(score, threshold, COMPACT_RISK_CELLS)
  const bar = 'risk '.length + textWidth(segments)
  const bound = ` < ${fixed(threshold)}`
  const spare = columns - COMPACT_LEAD.length - 2 - bar - ` ${fixed(score)}`.length - more.length
  const isBound = subject.length + bound.length <= spare
  const part = clip(subject, Math.max(1, spare))
  return (
    <Text>
      {brand(elements)}
      <Text color={LOOKS.allowed.color}>{' allowed'}</Text>
      {`  ${part}${more}  `}
      <Text dimColor>risk </Text>
      {meterCells(elements, segments, 'success')}
      {` ${fixed(score)}`}
      {isBound ? <Text dimColor>{bound}</Text> : null}
    </Text>
  )
}

// What a row shows of a decision under `display`: the full card or line, the one-line form, or nothing.
export function rowView(elements: Elements, described: Described, columns: number, narrow: Described, display: Display) {
  if (display === 'verbose' || described.weight === 'act') return inlineView(elements, described, columns, narrow)
  if (display === 'quiet' || described.weight === 'quiet') return null
  // The plain words go first when the row is short: a cut word reads worse than none.
  const fits = `${BRAND} ${described.badge}  ${described.line}`.length <= columns
  return badgeLine(elements, { ...described, headline: fits ? described.line : described.headline, detail: '' }, columns)
}

// A card for a decision worth noticing, a compact line for an allowed command, the one-line form where the row is narrow.
export function inlineView(elements: Elements, described: Described, columns: number, narrow: Described = described) {
  if (columns < MIN_CARD_COLUMNS) return badgeLine(elements, narrow, columns)
  const { inline } = described
  return inline.kind === 'card'
    ? card(elements, described, inline, columns)
    : compactLine(elements, described, inline.score, inline.threshold, columns)
}

// Each item behind a pixel in its app's color.
function tallyText({ Text }: Elements, items: readonly TallyItem[]) {
  return items.flatMap((item, index) => [index === 0 ? '' : '  ', <Text color={item.color}>{TALLY_DOT}</Text>, ` ${item.text}`])
}

// The whole items that fit `columns`, after a lead of `used` columns.
function itemsWithin(items: readonly TallyItem[], columns: number, used: number) {
  const kept: TallyItem[] = []
  for (const item of items) {
    if (used + tallyWidth([...kept, item]) > columns && kept.length > 0) break
    kept.push(item)
  }
  return kept
}

const BAND_LEAD = `${BRAND}  `

// Offline the decision model's checks step aside, so the band says which: only guard's fixed rules still hold.
// Offline, an app set to wait until done holds (the output, the command) instead of stepping aside.
export type Offline = { notChecking: readonly App[]; holding: readonly App[] }

const offlineItem = ({ notChecking, holding }: Offline): TallyItem => ({
  color: 'warning',
  text: [
    'model offline',
    ...(notChecking.length > 0 ? [`${notChecking.join(', ')} not checking`] : []),
    ...(holding.length > 0 ? [`${holding.join(', ')} holding`] : []),
  ].join(' · '),
})

// The tally above the prompt: the mark, then what vSR did, most urgent first.
export function band(elements: Elements, list: readonly Decision[], columns: number, isOffline: boolean, offline: Offline = { notChecking: [], holding: [] }) {
  const { Text } = elements
  const items = isOffline ? [offlineItem(offline), ...tallyOf(list)] : tallyOf(list)
  return (
    <Text wrap="truncate-end">
      {brand(elements)}
      {'  '}
      {tallyText(elements, itemsWithin(items, columns, BAND_LEAD.length))}
    </Text>
  )
}

// What the pane's foot says of the setup: where the classifier runs, the lines in force, the last calibration (the
// controls at its top show the apps and the waits).
// `spend` is what routing cost against the session's model; `labeled`, the answers given this session.
export type Overview = { settings: Settings; calibration: Calibration; isOffline: boolean; spend: Spend; labeled: number }

const dollars = (value: number) => (value < 0.01 ? '<$0.01' : `$${value.toFixed(2)}`)

// What routing came to: an estimate, at list price, of the same tokens on the session's model.
function spendWords({ spend }: Overview) {
  const { sessionModel } = spend
  if (spend.turns === 0 || sessionModel === undefined || spend.baseline === 0) return undefined
  const change = 1 - spend.actual / spend.baseline
  const verdict = change >= 0 ? `saved ${Math.round(change * 100)}%` : `cost ${Math.round(-change * 100)}% more`
  return `${dollars(spend.actual)} for ${spend.turns} subagent ${spend.turns === 1 ? 'turn' : 'turns'} vs ${dollars(spend.baseline)} on ${modelName(sessionModel)} · ${verdict} · est. at list price`
}

const tokenWords = ({ url, token }: Settings['runtime']) =>
  token === '' ? '' : isCleartextRemote(url) ? ' · token set, sent unencrypted' : ' · token set'

function overviewRows(about: Overview) {
  const { settings: { runtime, thresholds: t, log }, calibration, isOffline, labeled } = about
  const routing = spendWords(about)
  const since = `${calibration.since} ${calibration.since === 1 ? 'decision' : 'decisions'} since`
  return [
    ...(routing === undefined ? [] : [['routing', routing] as const]),
    ['model', `${runtime.kind} · ${runtime.url}${tokenWords(runtime)}${isOffline ? ' · not answering' : ''}`],
    [
      'router',
      `easy below ${fixed(t.router.easyBelow)}, hard from ${fixed(t.router.hardFrom)} (of 2), sure from ${fixed(t.router.minConfidence)} · reasoning steps few below ${fixed(t.router.stepsLow)}, many from ${fixed(t.router.stepsHigh)}`,
    ],
    [
      'thresholds',
      [
        `guard holds at ${fixed(t.guard.review)}`,
        `check flags at ${fixed(t.check.unsupported)}`,
        `shield warns at ${fixed(t.shield.inject)}`,
        `redact hides at ${fixed(t.redact.pii)}`,
        `escalate at ${fixed(t.escalate.feedback)}`,
      ].join(' · '),
    ],
    ['calibrated', log.path === '' ? 'log off' : `${calibration.at ?? 'never'} · ${since}${labeled > 0 ? ` · ${labeled} answered here` : ''} · log ${log.path}`],
  ] as const
}

export function overview(elements: Elements, about: Overview) {
  const { Box, Text } = elements
  return (
    <Box flexDirection="column" marginTop={1}>
      {overviewRows(about).map(([label, value]) => (
        <Text>
          <Text dimColor>{`${label}  `}</Text>
          {value}
        </Text>
      ))}
    </Box>
  )
}

// The rows the overview takes at `columns`, its gap above included.
export const overviewHeight = (about: Overview, columns: number) =>
  1 + overviewRows(about).reduce((sum, [label, value]) => sum + Math.max(1, Math.ceil((label.length + 2 + value.length) / columns)), 0)

// What pressing a control at the top of the pane does: turn an app on or off, or pick how long one waits.
export type Controls = { toggle: (app: App) => void; wait: (app: WaitApp, choice: WaitChoice) => void }

// Three apps to a row: with each hotkey drawn before its name (`1: ● guard`), six would not fit 60 columns.
const APPS_PER_ROW = 3
const CONTROL_LABEL_WIDTH = 'redact waits'.length + 2
const WAIT_WORDS: Record<WaitChoice, string> = { '1': '1s', '2': '2s', '3': '3s', done: 'until done' }
const APP_ROWS = Array.from({ length: Math.ceil(APP_NAMES.length / APPS_PER_ROW) }, (_, row) =>
  APP_NAMES.slice(row * APPS_PER_ROW, (row + 1) * APPS_PER_ROW),
)

// The rows the controls take, their gap below included.
export const CONTROLS_HEIGHT = APP_ROWS.length + WAIT_APPS.length + 1

// The apps, each pressed (or its digit typed) to turn it on or off, then how long each protective app waits: the
// chosen wait a pill in the app's colour, the rest dim and as wide, so the row does not shift when the pick moves.
export function controls(elements: Elements, settings: Settings, act: Controls) {
  const { Box, Button, Text } = elements
  return (
    <Box flexDirection="column" marginBottom={1}>
      {APP_ROWS.map((row, index) => (
        <Box flexDirection="row">
          <Text>{(index === 0 ? 'apps' : '').padEnd(CONTROL_LABEL_WIDTH)}</Text>
          <Box flexDirection="row" gap={2}>
            {row.map(app => {
              const isOn = settings.apps[app]
              return (
                <Button key={`toggle:${app}`} plain hotkey={String(APP_NAMES.indexOf(app) + 1)} dimColor={!isOn} onPress={() => act.toggle(app)}>
                  {isOn ? <Text color={APPS[app].color}>{`● ${app}`}</Text> : `○ ${app}`}
                </Button>
              )
            })}
          </Box>
        </Box>
      ))}
      {WAIT_APPS.map(app => {
        const isOn = settings.apps[app]
        const chosen = choiceOf(settings.waits[app])
        return (
          <Box flexDirection="row">
            <Text dimColor={!isOn}>{`${app} waits`.padEnd(CONTROL_LABEL_WIDTH)}</Text>
            <Box flexDirection="row" gap={1}>
              {WAIT_CHOICES.map(choice => {
                const words = ` ${WAIT_WORDS[choice]} `
                return (
                  <Button key={`wait:${app}:${choice}`} plain dimColor={!isOn || choice !== chosen} onPress={() => act.wait(app, choice)}>
                    {choice === chosen ? (
                      <Text backgroundColor={APPS[app].color} color="inverseText" dimColor={!isOn}>
                        {words}
                      </Text>
                    ) : (
                      words
                    )}
                  </Button>
                )
              })}
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}

type Entry = { decision: Decision; described: Described; second: string; rows: number }

// The answers given in the pane, and what pressing one does; absent where answers cannot be kept (the log is off).
export type Labeling = { given: Readonly<Record<string, Answer>>; answer: (decision: Decision, answer: Answer) => void }

// Each section gets its header and newest entry first, then older entries go round the sections while rows are left.
function fitSections(sections: { section: Section; entries: Entry[] }[], budget: number) {
  const shown = sections.map(() => 0)
  let used = sections.reduce((sum, { entries }) => sum + 1 + entries[0].rows, 0)
  sections.forEach((_, index) => (shown[index] = 1))
  for (let isGrowing = true; isGrowing; ) {
    isGrowing = false
    sections.forEach(({ entries }, index) => {
      const next = entries[shown[index]]
      if (next !== undefined && used + next.rows <= budget) {
        shown[index] += 1
        used += next.rows
        isGrowing = true
      }
    })
  }
  return sections.map(({ section, entries }, index) => ({ section, count: entries.length, entries: entries.slice(0, shown[index]) }))
}

const WEIGHT_ORDER: Record<Weight, number> = { act: 0, notice: 1, quiet: 2 }

// Said once above the answers, so a person knows what answering does.
const ANSWER_HINT = 'Answers tune the thresholds when you run /vsr:calibrate.'

export function dashboard(
  elements: Elements,
  logo: Logo,
  list: readonly Decision[],
  columns: number,
  rows: number,
  about: Overview,
  act: Controls,
  labeling?: Labeling,
) {
  const { Box, Button, Text } = elements
  const tally = tallyOf(list)
  const tallyRows = Math.max(1, Math.ceil(tallyWidth(tally) / columns))
  const hint = labeling !== undefined && list.some(decision => answersFor(decision) !== undefined)
  const sections = SECTIONS.flatMap(section => {
    const entries = list
      .filter(decision => sectionOf(decision) === section)
      .reverse()
      .map(decision => {
        const described = describe(decision)
        const second = `${described.headline} · ${described.detail}`
        const answers = labeling && answersFor(decision) ? 1 : 0
        return { decision, described, second, rows: 1 + Math.ceil(second.length / columns) + answers }
      })
      // What needed a look (a held command, a withheld output) before what only went on, each newest first, so a
      // run of ALLOWED rows does not push a HELD one out of view.
      .sort((a, b) => WEIGHT_ORDER[a.described.weight] - WEIGHT_ORDER[b.described.weight])
    return entries.length === 0 ? [] : [{ section, entries }]
  })
  return (
    <Box flexDirection="column">
      {wordmark(elements, logo)}
      {controls(elements, about.settings, act)}
      <Text>{tallyText(elements, tally)}</Text>
      {hint ? <Text dimColor>{ANSWER_HINT}</Text> : null}
      {fitSections(sections, rows - wordmarkHeight(logo) - CONTROLS_HEIGHT - tallyRows - (hint ? Math.ceil(ANSWER_HINT.length / columns) : 0) - overviewHeight(about, columns)).map(({ section, count, entries }) => (
        <Box flexDirection="column">
          <Text>
            <Text color={APPS[section].color}>{TALLY_DOT}</Text>
            <Text weight="bold">{` ${APPS[section].name} · ${count}`}</Text>
          </Text>
          {entries.map(({ decision, described: { badge, color, subject, more = '', headline, detail, weight } }) => {
            const kind = labeling ? answersFor(decision) : undefined
            return (
              <Box flexDirection="column">
                <Text>
                  {weight === 'act' ? pill(elements, badge, color) : <Text color={color}>{` ${badge.toLowerCase()} `}</Text>}
                  {` ${clip(subject, Math.max(1, columns - badge.length - 3 - more.length))}${more}`}
                </Text>
                <Text>
                  {headline}
                  {detail === '' ? null : <Text dimColor>{` · ${detail}`}</Text>}
                </Text>
                {labeling && kind ? (
                  <Box flexDirection="row" gap={1}>
                    {ANSWERS[kind].map(({ answer, label }) => {
                      const isChosen = labeling.given[decision.id] === answer
                      return (
                        <Button
                          key={`${decision.id}:${answer}`}
                          label={isChosen ? `✓ ${label}` : label}
                          variant={isChosen ? 'primary' : 'secondary'}
                          plain
                          onPress={() => labeling.answer(decision, answer)}
                        >
                          {isChosen ? (
                            <Text backgroundColor="suggestion" color="inverseText">{` ✓ ${label} `}</Text>
                          ) : (
                            <Text dimColor>
                              {' '}
                              <Text dimColor underline>
                                {label}
                              </Text>{' '}
                            </Text>
                          )}
                        </Button>
                      )
                    })}
                  </Box>
                ) : null}
              </Box>
            )
          })}
        </Box>
      ))}
      {overview(elements, about)}
    </Box>
  )
}
