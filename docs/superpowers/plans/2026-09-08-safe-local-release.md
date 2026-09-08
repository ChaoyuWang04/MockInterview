# Safe Local Release Implementation Plan

> **For agentic workers:** Follow this plan task-by-task. Use test-driven development for behavior changes and preserve the approved scope.

**Goal:** Separate verification builds from live releases and make local production rebuilds switch between immutable blue/green slots with health checks and rollback.

**Architecture:** `npm run build` writes `.next-check`. `leet-rebuild` builds the inactive `.next-blue` or `.next-green` slot while the active server remains online, then switches, verifies HTML/CSS, and rolls back on failure. The daemon records the exact dist directory used by the process.

**Tech Stack:** Bash, Python 3, Next.js 15, Vitest, curl

---

### Task 1: Add failing release-contract tests

**Files:**
- Create: `tests/service-operations.test.ts`

- [ ] Assert ordinary build targets `.next-check`, production release uses blue/green slots, the daemon receives an explicit dist directory, and active/server state is tracked atomically.
- [ ] Cover fresh install, legacy migration, actual-slot-first candidate selection, illegal/missing state, operation locking, and PID ownership fail-closed behavior.
- [ ] Run the focused test and confirm it fails against the current scripts.

### Task 2: Implement isolated builds and blue/green switching

**Files:**
- Modify: `package.json`
- Modify: `scripts/leet-server.sh`
- Modify: `scripts/leet-daemon.py`
- Modify: `.gitignore`
- Modify: `eslint.config.mjs`
- Modify: `tsconfig.json`
- Modify: `next.config.ts`

- [ ] Make `npm run build` target `.next-check` and add the internal release build entry.
- [ ] Add validated active/server dist state, actual-slot-first inactive selection, an atomic operation lock with stale-lock recovery, and atomic state writes.
- [ ] Verify PID ownership by project directory, Next command, and port; never kill an unknown port owner.
- [ ] Add bounded homepage plus CSS/JS health checks, a sub-10-second normal switch, and rollback whose own health result is reported truthfully.
- [ ] Pass the selected dist directory into the daemon instead of hard-coding `.next-prod`.
- [ ] Route `npm start` through the controlled service command and split build/runtime logs.
- [ ] Ignore and type-register the three fixed build directories.
- [ ] Run the focused test until green.

### Task 3: Update every active workflow document

**Files:**
- Modify: `README.md`
- Modify: `docs/08-常驻服务.md`
- Modify: `docs/09-日常维护.md`
- Modify: `docs/10-基模报告流程.md`

- [ ] Define `build`, `restart`, and `rebuild` precisely.
- [ ] Replace every active instruction that could imply building into the live directory.
- [ ] Document automatic health checking, rollback, build-slot status, and the correct action for Markdown, `public/`, and code changes.
- [ ] Leave historical `docs/superpowers/` artifacts unchanged except this new approved design and plan.

### Task 4: Verify the real failure mode and recovery

- [ ] Run the focused test and full `npm test`.
- [ ] Run `npm run build` and confirm it writes `.next-check` without changing the active production BUILD_ID.
- [ ] Run `leet-restart` to align the current legacy process, then `leet-rebuild` to migrate to a blue/green slot.
- [ ] Request the homepage and every linked same-origin CSS/JS asset; require HTTP 200.
- [ ] Run a second isolated `npm run build` while production stays live and repeat the HTTP/CSS check.
- [ ] Force a candidate-build failure and confirm the live PID, actual slot, BUILD_ID, HTML, CSS, and JS remain unchanged.
- [ ] Run `npm run build`, full tests, `git diff --check`, and inspect the final diff.
- [ ] Commit as one scoped operational fix.
