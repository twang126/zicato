// test/operator_provenance.test.mjs — OPERATOR CANDIDATES on the dashboard (issue #570).
//
// A candidate that `zicato proposer submit` queued carries an `operator`
// record in experiment.json, and the ledger row carries `source` + `author`.
//
// Pins:
//   * the dossier's proposal card shows ONE faint caption line naming the
//     author, the edited generation and any promoted change it takes back;
//     a proposer's card has no such line;
//   * the ledger row shows a faint caption under the idea for an operator's
//     row only;
//   * both digests change when the provenance changes and stay identical on
//     a no-op beat (G10: digest-gated rendering).

import { installDom, test, run, assert, assertEqual, assertDeep } from './harness.mjs';

installDom();

const cand = await import('../js/views/candidate.js');
const ledgerMod = await import('../js/views/ledger.js');

function classOf(node) { return (node && node.getAttribute && node.getAttribute('class')) || ''; }
function hasClass(node, cls) { return classOf(node).split(/\s+/).includes(cls); }
function allByClass(host, cls) {
  return host.querySelectorAll('[class]').filter((n) => hasClass(n, cls));
}
function mountInto(node) { const h = document.createElement('div'); if (node) h.appendChild(node); return h; }

function expFixture(operator) {
  return {
    generation_id: 'v4',
    parent_generation_id: 'v3',
    hypothesis: { core_idea: 'Greet with more warmth', why: 'friendlier', modulating: ['greet_logic'] },
    patches: { greet_logic: { mutation_id: 'greet_logic', op: 'replace', new_content: 'x' } },
    ...(operator ? { operator } : {}),
  };
}

const OPERATOR = {
  author: 'alice', based_on: 'v1', queue_id: 'q1', submitted_at: '2026-10-13T00:00:00Z',
  holdout_access: false, reverted_points: ['greeting'],
};

test('buildProposalModel: lifts the operator provenance, and a proposer has none', () => {
  assertDeep(cand.buildProposalModel(expFixture(OPERATOR)).operator,
    { author: 'alice', basedOn: 'v1', revertedPoints: ['greeting'] }, 'the operator record is lifted');
  assertEqual(cand.buildProposalModel(expFixture(null)).operator, null, 'a proposer candidate has no operator');
});

test('proposalHeader: one faint caption line for an operator, none for a proposer', () => {
  const op = mountInto(cand.proposalHeader(cand.buildProposalModel(expFixture(OPERATOR))));
  const lines = allByClass(op, 'dn-proposal-source');
  assertEqual(lines.length, 1, 'exactly one caption line');
  assert(hasClass(lines[0], 'dn-faint'), 'the caption is faint metadata, never a tag');
  assertEqual(lines[0].textContent,
    'submitted by an operator · alice · edited v1 · takes back the promoted change at greeting');
  const proposer = mountInto(cand.proposalHeader(cand.buildProposalModel(expFixture(null))));
  assertEqual(allByClass(proposer, 'dn-proposal-source').length, 0, 'a proposer card has no caption');
});

test('proposalDigest: the provenance is folded in; a no-op beat is identical', () => {
  const digest = (op) => JSON.stringify(cand.proposalDigest(cand.buildProposalModel(expFixture(op))));
  assertEqual(digest(OPERATOR), digest({ ...OPERATOR }), 'a no-op beat is byte-identical');
  assert(digest(OPERATOR) !== digest(null), 'an operator record changes the digest');
  assert(digest(OPERATOR) !== digest({ ...OPERATOR, reverted_points: [] }), 'reverted points change it');
});

function ledgerFixture(source, author) {
  return {
    epoch_id: 'e0',
    experiments: [{
      generation_id: 'v4', parent_generation_id: 'v3', round_index: 2, core_idea: 'Greet with more warmth',
      mutation_ids: ['greet_logic'], decision: 'promoted', decision_label: 'promoted', promoted: true,
      rejection_reason: null, scalar_score_delta: -0.2, drift_loss_delta: -0.1, pass_rate_delta: 0,
      source, author,
    }],
  };
}

function mountLedger(ledger) {
  ledgerMod._resetLedgerExpansion();
  return mountInto(ledgerMod.buildExperimentsLedger(ledger, { epochId: 'e0' }));
}

test('ledger: an operator row has one faint caption under its idea; a proposer row has none', () => {
  const op = allByClass(mountLedger(ledgerFixture('operator', 'alice')), 'dn-ledger-source');
  assertEqual(op.length, 1, 'the operator row carries one caption');
  assertEqual(op[0].textContent, 'operator · alice');
  assert(hasClass(op[0], 'dn-faint'), 'the caption is faint metadata');
  assertEqual(allByClass(mountLedger(ledgerFixture('proposer', null)), 'dn-ledger-source').length, 0,
    'a proposer row is unchanged');
});

test('ledgerDigest: the source and author are folded in', () => {
  const digest = (s, a) => JSON.stringify(ledgerMod.ledgerDigest(ledgerFixture(s, a)));
  assertEqual(digest('operator', 'alice'), digest('operator', 'alice'), 'a no-op beat is byte-identical');
  assert(digest('operator', 'alice') !== digest('proposer', null), 'the source changes the digest');
  assert(digest('operator', 'alice') !== digest('operator', 'bob'), 'the author changes the digest');
});

run();
