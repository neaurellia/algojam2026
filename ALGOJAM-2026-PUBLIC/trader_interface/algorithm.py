import math

import numpy as np


class Algorithm():

    # Instruments this algorithm takes a position on. Anything not listed here
    # is held flat.
    ENABLED = [
        "UQ Dollar",
        "Sausage Sizzle",
        "Boat Party Ticket",
        "Fintech Token",
        "Thrifted Jeans",
        "Sausage",
        "Bread",
        "MenuDash",
    ]

    # Order in which instruments are trimmed when the daily budget gets tight:
    # last entries are trimmed first.
    PRIORITY = [
        "Sausage Sizzle",
        "UQ Dollar",
        "Fintech Token",
        "Boat Party Ticket",
        "Bread",
        "Sausage",
        "Thrifted Jeans",
        "MenuDash",
        "Liferaft Ticket",
    ]

    # Stay this far under the exchange's $600K cap. A breach zeroes EVERY
    # position for the day, not just the offending one.
    BUDGET_CEILING = 580_000

    # ---- UQ Dollar ----------------------------------------------------------
    # Anchor is a rolling mean rather than a hardcoded $100 peg, so it
    # survives a repricing instead of betting the peg never moves.
    UQ_ANCHOR_WINDOW = 90

    # ---- Boat Party Ticket --------------------------------------------------
    # Averaged across four windows rather than committing to one.
    BOAT_WINDOWS = (3, 5, 10, 20)

    # ---- Fintech Token ------------------------------------------------------
    FINTECH_WINDOW = 5

    # Volatility filter: go flat after an outsized single-day move, resume once
    # calm.
    FT_JUMP_PAUSE = True
    FT_JUMP_SIGMA = 3.0        # a move over N rolling std devs starts a pause
    FT_CALM_DAYS = 2           # consecutive quiet days needed to resume
    FT_VOL_WINDOW = 20

    # ---- Sausage Sizzle -----------------------------------------------------
    # Coefficients are refit daily on past data only (see sizzle_coefficients),
    # rather than a single OLS fit over the whole series applied from day 0.
    # Only the SIGN of the prediction is traded, so what matters is the ratio
    # between the coefficients, not their scale.
    SIZZLE_ROLLING = True
    SIZZLE_WINDOW = 60         # None = expanding; int = trailing window
    SIZZLE_MIN_SAMPLE = 20     # stay flat until the fit has this many days
    # const/Bread/Sausage below are the full-sample (non-causal) fit, used
    # only when SIZZLE_ROLLING is False.
    #
    # MenuDash is NOT refit daily -- there is no causal rolling equivalent --
    # so its coefficient and halflife below are always what's traded, in both
    # branches. It enters SMOOTHED (EWMA), not raw; see sizzle_menudash_term.
    SIZZLE_COEFFS = {"const": 0.0053, "Bread": 0.0769,
                     "Sausage": 1.6780, "MenuDash": 4.2432}
    SIZZLE_MENUDASH_HALFLIFE = 5

    # ---- Thrifted Jeans -----------------------------------------------------
    # MA crossover, traded only while the slow MA is actually going somewhere.
    TJ_FAST_WINDOW = 3
    TJ_SLOW_WINDOW = 15
    TJ_SLOPE_LOOKBACK = 10     # compare the slow MA against N days ago
    TJ_MIN_SLOPE = 0.05        # need >5% change over that span to trade

    # ---- Sausage ------------------------------------------------------------
    # Donchian breakout.
    SAUSAGE_WINDOW = 8

    # ---- Bread --------------------------------------------------------------
    # Momentum against a week ago. Flip BREAD_DONCHIAN to trade the Donchian
    # alternative instead.
    BREAD_LOOKBACK = 7
    BREAD_DONCHIAN = False
    BREAD_WINDOW = 8

    # ---- MenuDash -----------------------------------------------------------
    # Short-window z-score with a deadband.
    MENUDASH_WINDOW = 10
    MENUDASH_THRESHOLD = 0.5

    # Human-readable description of each signal, keyed by instrument. report.py
    # prints this straight out, so the documented settings can never drift from
    # the ones actually traded. "params" counts values fitted to the data.
    SPEC = {
        "UQ Dollar":         ("w = 90",    "mean(P[-90:]) - P[-1]",          1),
        "Sausage Sizzle":    ("daily refit, w=60", "OLS on Bread/Sausage + smoothed MenuDash, sign traded", 3),
        "Boat Party Ticket": ("ensemble",  "sign vote, w in 3/5/10/20",      0),
        "Fintech Token":     ("w = 5",     "mean(P[-5:]) - P[-1] + vol gate", 1),
        "Thrifted Jeans":    ("MA 3/15",   "fast-slow, slope>5% over 10d",   4),
        "Sausage":           ("w = 8",     "Donchian breakout, held between", 1),
        "Bread":             ("k = 7",     "P[-1] - P[-8]  (momentum)",      1),
        "MenuDash":          ("w = 10",    "-(z-score), |z| > 0.5",          2),
    }

    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        self.data = {}  # Historical data of all instruments
        self.positionLimits = {}    # Initialise position limits
        self.day = 0     # Initialise the current day as 0
        self.positions = positions   # Initialise the current positions
        # Fintech volatility-filter state, and the Donchian directions, all of
        # which persist across days because the engine reuses this object.
        self.ft_paused = False
        self.ft_calm_run = 0
        self.sausage_direction = 0
        self.bread_direction = 0

    def get_current_price(self, instrument):
        """
        Helper function to fetch current price of an instrument.
        """
        return self.data[instrument][-1]

    def reversion_signal(self, instrument, window):
        """
        Positive when price sits below its recent mean, i.e. when we expect it
        to rise. Returns None until there is enough history.
        """
        prices = self.data[instrument]
        if len(prices) < window:
            return None
        recent = np.array(prices[-window:], dtype=float)
        return recent.mean() - recent[-1]

    def ensemble_reversion_signal(self, instrument, windows):
        """
        Majority vote of the reversion signal across several windows. Each
        window contributes only its DIRECTION, not its magnitude, so the
        longest window can't dominate by sitting further from today's price.
        """
        votes = []
        for window in windows:
            signal = self.reversion_signal(instrument, window)
            if signal is not None:
                votes.append(np.sign(signal))
        if not votes:
            return None
        return float(sum(votes) / len(votes))

    def donchian_signal(self, instrument, window, attribute):
        """
        Breakout: a new N-day high flips us long, a new N-day low flips us
        short, and between the two extremes we HOLD the last direction rather
        than re-deciding, which is what stops it churning on noise. Returns 0
        until the first breakout, i.e. flat.
        """
        prices = self.data[instrument]
        if len(prices) < window:
            return None
        recent = np.array(prices[-window:], dtype=float)
        if prices[-1] >= recent.max():
            setattr(self, attribute, 1)
        elif prices[-1] <= recent.min():
            setattr(self, attribute, -1)
        return float(getattr(self, attribute))

    def uq_signal(self):
        """
        Positive when UQ Dollar sits below its own rolling mean, i.e. when the
        peg should pull it back up.

        Deliberately NOT reversion_signal, which returns None until its window
        fills and would sit flat for the first 90 days. Slicing a short list
        just yields an expanding mean, so the anchor is usable from day 0.
        """
        recent = np.array(self.data["UQ Dollar"][-self.UQ_ANCHOR_WINDOW:],
                          dtype=float)
        return recent.mean() - recent[-1]

    def sizzle_inputs(self):
        """Today's moves in the two cost inputs, or None if unavailable."""
        moves = []
        for name in ("Bread", "Sausage"):
            prices = self.data.get(name)
            if prices is None or len(prices) < 2:
                return None
            moves.append(prices[-1] - prices[-2])
        return moves

    def sizzle_coefficients(self):
        """
        Least-squares fit of tomorrow's Sizzle move on today's input moves,
        using only days that have already happened.

        Training pairs are (input moves on day k, Sizzle move on day k+1) for
        every k where both sides are known. Today's own input move is
        deliberately NOT a training row: the Sizzle move it predicts has not
        happened yet, and including it would be a one-day lookahead.

        Returns None until SIZZLE_MIN_SAMPLE pairs exist, or if the inputs are
        collinear enough that the fit is not defined.
        """
        series = {}
        for name in ("Sausage Sizzle", "Bread", "Sausage"):
            prices = self.data.get(name)
            if prices is None or len(prices) < 3:
                return None
            series[name] = np.diff(np.asarray(prices, dtype=float))

        # diff[i] is the move INTO day i+1, so dropping the last input row and
        # the first target row lines each input move up with the Sizzle move
        # one day after it.
        targets = series["Sausage Sizzle"][1:]
        features = np.column_stack([
            series["Bread"][:-1],
            series["Sausage"][:-1],
            np.ones(len(targets)),
        ])

        if self.SIZZLE_WINDOW is not None:
            targets = targets[-self.SIZZLE_WINDOW:]
            features = features[-self.SIZZLE_WINDOW:]

        if len(targets) < self.SIZZLE_MIN_SAMPLE:
            return None

        solution, _, rank, _ = np.linalg.lstsq(features, targets, rcond=None)
        if rank < features.shape[1]:
            # Underdetermined: lstsq still returns a vector, but it is the
            # minimum-norm one rather than a fit, so it says nothing.
            return None
        return solution

    def sizzle_menudash_term(self):
        """
        Smoothed-MenuDash contribution, added on top of the Bread/Sausage
        prediction in sizzle_signal.

        The EWMA is recomputed over full history each call and only its
        one-day CHANGE is used, so this returns 0.0 rather than None when
        history is short: it is a bolt-on term, not a gate on the signal.
        """
        menudash = self.data.get("MenuDash")
        if menudash is None or len(menudash) < 2:
            return 0.0
        alpha = 1 - 0.5 ** (1.0 / self.SIZZLE_MENUDASH_HALFLIFE)
        ewma = menudash[0]
        previous = ewma
        for price in menudash[1:]:
            previous = ewma
            ewma = alpha * price + (1 - alpha) * ewma
        return self.SIZZLE_COEFFS["MenuDash"] * (ewma - previous)

    def sizzle_signal(self):
        """
        Predict tomorrow's move in Sausage Sizzle from today's moves in Bread
        and Sausage, plus a smoothed-MenuDash term.

        The Bread/Sausage coefficients come from a daily refit on a trailing
        SIZZLE_WINDOW-day window of past data. Returns None while the fit is
        still too short to trust, which keeps us flat rather than falling
        back on the static SIZZLE_COEFFS.
        """
        moves = self.sizzle_inputs()
        if moves is None:
            return None
        d_bread, d_sausage = moves

        if not self.SIZZLE_ROLLING:
            prediction = (self.SIZZLE_COEFFS["const"]
                          + self.SIZZLE_COEFFS["Bread"] * d_bread
                          + self.SIZZLE_COEFFS["Sausage"] * d_sausage)
        else:
            coefficients = self.sizzle_coefficients()
            if coefficients is None:
                return None
            b_bread, b_sausage, const = coefficients
            prediction = const + b_bread * d_bread + b_sausage * d_sausage

        return prediction + self.sizzle_menudash_term()

    def jeans_signal(self):
        """
        MA crossover gated by the slow MA's slope. Returns the fast-slow gap
        while the slow MA is trending, 0 while it is flat (stay out of the
        chop), and None until there is enough history for both.
        """
        prices = self.data["Thrifted Jeans"]
        span = self.TJ_SLOW_WINDOW + self.TJ_SLOPE_LOOKBACK
        if len(prices) < span:
            return None

        recent = np.array(prices, dtype=float)
        fast = recent[-self.TJ_FAST_WINDOW:].mean()
        slow = recent[-self.TJ_SLOW_WINDOW:].mean()
        # The slow MA as it stood TJ_SLOPE_LOOKBACK days ago.
        previous_slow = recent[-span:-self.TJ_SLOPE_LOOKBACK].mean()
        if previous_slow == 0:
            return None

        slope = (slow - previous_slow) / previous_slow
        if abs(slope) < self.TJ_MIN_SLOPE:
            return 0.0
        return fast - slow

    def bread_signal(self):
        """Momentum against a week ago, or Donchian if BREAD_DONCHIAN is set."""
        if self.BREAD_DONCHIAN:
            return self.donchian_signal("Bread", self.BREAD_WINDOW,
                                        "bread_direction")
        prices = self.data["Bread"]
        if len(prices) <= self.BREAD_LOOKBACK:
            return None
        return prices[-1] - prices[-(self.BREAD_LOOKBACK + 1)]

    def menudash_signal(self):
        """
        Positive when MenuDash sits below its own recent mean, i.e. when we
        expect the posted price to drift back up. Flat when the deviation is
        too small to be worth trading. Returns None until there is enough
        history.
        """
        prices = self.data["MenuDash"]
        if len(prices) < self.MENUDASH_WINDOW:
            return None
        recent = np.array(prices[-self.MENUDASH_WINDOW:], dtype=float)
        sd = recent.std()
        if sd == 0:
            return None
        z = (recent[-1] - recent.mean()) / sd
        if abs(z) < self.MENUDASH_THRESHOLD:
            return 0.0
        # Above the mean -> expect a fall -> negative signal -> short
        return -z

    def fintech_is_paused(self):
        """
        Track whether an outsized move has put Fintech on the sidelines.

        A move larger than FT_JUMP_SIGMA times its own recent volatility means
        the level has changed, so the rolling mean is now an average of two
        different regimes and cannot be reverted to. Sit out until the price
        has been quiet for FT_CALM_DAYS in a row; any loud day resets that
        count.
        """
        prices = self.data["Fintech Token"]
        if len(prices) < self.FT_VOL_WINDOW + 1:
            return False

        recent = np.array(prices[-(self.FT_VOL_WINDOW + 1):], dtype=float)
        moves = np.diff(recent)
        # Exclude today's move from its own volatility estimate, or a big jump
        # inflates the denominator and hides itself.
        volatility = moves[:-1].std()
        if volatility == 0:
            return self.ft_paused

        magnitude = abs(moves[-1]) / volatility
        if magnitude > self.FT_JUMP_SIGMA:
            self.ft_paused = True
            self.ft_calm_run = 0
        elif self.ft_paused:
            if magnitude < 1.0:
                self.ft_calm_run += 1
                if self.ft_calm_run >= self.FT_CALM_DAYS:
                    self.ft_paused = False
                    self.ft_calm_run = 0
            else:
                self.ft_calm_run = 0
        return self.ft_paused

    def get_signal(self, instrument):
        """Signal for one instrument; sign is the direction we want to hold."""
        if instrument == "UQ Dollar":
            return self.uq_signal()
        if instrument == "Sausage Sizzle":
            return self.sizzle_signal()
        if instrument == "Thrifted Jeans":
            return self.jeans_signal()
        if instrument == "Boat Party Ticket":
            return self.ensemble_reversion_signal(instrument, self.BOAT_WINDOWS)
        if instrument == "Sausage":
            return self.donchian_signal("Sausage", self.SAUSAGE_WINDOW,
                                        "sausage_direction")
        if instrument == "Bread":
            return self.bread_signal()
        if instrument == "MenuDash":
            return self.menudash_signal()
        if instrument == "Fintech Token":
            # State has to advance every day, so update the filter before
            # deciding whether to act on it.
            if self.FT_JUMP_PAUSE and self.fintech_is_paused():
                return 0.0
            return self.reversion_signal(instrument, self.FINTECH_WINDOW)
        return None

    def enforce_budget(self, desiredPositions):
        """
        Trim positions, cheapest-to-lose first, until the day's total exposure
        fits under BUDGET_CEILING.

        Walks reverse PRIORITY and SHRINKS each position by just enough units
        to close the gap, only zeroing an instrument when trimming it whole
        still is not enough.
        """
        def exposure():
            return sum(abs(position) * self.data[instrument][-1]
                       for instrument, position in desiredPositions.items()
                       if position)

        total = exposure()
        if total <= self.BUDGET_CEILING:
            return desiredPositions

        # PRIORITY first, then anything it forgot. Without that tail an
        # instrument added to ENABLED but missing from PRIORITY could never be
        # trimmed, the loop would end still over budget, and the ENGINE zeroes
        # EVERY position for the day -- far worse than any trim.
        order = list(reversed(self.PRIORITY))
        order += [name for name in desiredPositions if name not in self.PRIORITY]

        actions = []
        for instrument in order:
            if total <= self.BUDGET_CEILING:
                break
            position = desiredPositions.get(instrument)
            if not position:
                continue
            price = self.data[instrument][-1]
            if price <= 0:
                continue
            # ceil, so rounding always lands us UNDER the ceiling, never a
            # dollar over it.
            shed = math.ceil((total - self.BUDGET_CEILING) / price)
            if shed < abs(position):
                keep = abs(position) - shed
                desiredPositions[instrument] = keep if position > 0 else -keep
                actions.append(f"{instrument} {abs(position)}->{keep}")
            else:
                desiredPositions[instrument] = 0
                actions.append(f"{instrument} flat")
            total = exposure()
        print(f"BUDGET GUARD: {', '.join(actions)} -> ${total:,.0f}")
        return desiredPositions

    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        # Get current position
        currentPositions = self.positions
        # Get position limits
        positionLimits = self.positionLimits

        # Declare a store for desired positions
        desiredPositions = {}
        # Loop through all the instruments you can take positions on.
        for instrument, positionLimit in positionLimits.items():
            # For each instrument initilise desired position to zero
            desiredPositions[instrument] = 0

        # Display the current trading day
        print("Starting Algorithm for Day:", self.day)

        # Greedy sizing: full position limit in whichever direction the signal
        # points, then enforce_budget trims positions if the day does not
        # fit under the cap.
        for instrument in self.ENABLED:
            signal = self.get_signal(instrument)
            if signal is None or signal == 0:
                continue
            limit = positionLimits[instrument]
            desiredPositions[instrument] = limit if signal > 0 else -limit
            print(f"{instrument}: ${self.get_current_price(instrument)} "
                  f"signal={signal:+.4f} position={desiredPositions[instrument]}")

        desiredPositions = self.enforce_budget(desiredPositions)

        # Display the end of trading day
        print("Ending Algorithm for Day:", self.day, "\n")
        return desiredPositions
