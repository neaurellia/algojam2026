"""Refinement book: the combined signals with the overfitting taken out.

Differences from algorithm.py, all aimed at closing the fit/test gap rather
than raising the Round 1 headline:

  Boat Party    single w=5  ->  ensemble of w in {3,5,10,20}, removing the
                window parameter entirely. The single best window (w=3,
                $92,610) was chosen with hindsight; blind picking gets you
                the median ($24,890) and the ensemble beats that by 74%.
  Thrifted      (5,20,0.02) ->  (3,15,0.05). The old setting was the single
  Jeans         best of 45 grid combinations, so roughly $20K of its $81,664
                was selection bias. The new one is the most balanced on the
                grid: fit/test ratio 1.7x against the old 24.5x.
  Bread         flat        ->  k=7 momentum. Fit $11,070 / test $11,020,
                the evenest split in the book, and it was already costed.

Expected effect: Round 1 total drops from $369,468 to roughly $374K with
Bread included but with far less of it resting on hindsight. The
walk-forward estimate of $250-280K is the number to plan against.
"""
import numpy as np


class Algorithm():

    # Instruments this algorithm takes a position on. Anything not listed here
    # is held flat. Thrifted Jeans and Bread are parked: Jeans is the only
    # signal that fails the shuffle test (94.5%, p ~ 0.055) and earns 148% of
    # its money in its best 20 days, and Bread returns 36c per dollar of budget
    # tied up. Both are revisitable once the core four are clean.
    ENABLED = [
        "UQ Dollar",
        "Sausage Sizzle",
        "Boat Party Ticket",
        "Fintech Token",
    ]

    # Order in which instruments are kept when the daily budget gets tight:
    # last entries are dropped first. Ranked by how much each signal can be
    # relied on rather than by how much it earned -- Sizzle and UQ Dollar have
    # the highest captures and the evenest fit/test splits, Fintech has the
    # flattest parameter plateau. Liferaft is last because one unit is ~$150K
    # and it pays no local P&L at all.
    PRIORITY = [
        "Sausage Sizzle",
        "UQ Dollar",
        "Fintech Token",
        "Boat Party Ticket",
        "Bread",
        "Thrifted Jeans",
        "MenuDash",
        "Sausage",
        "Liferaft Ticket",
    ]

    # Stay this far under the exchange's $600K cap. The engine values positions
    # at the same prices used here, so without a margin a rounding edge could
    # tip the total over -- and a breach zeroes EVERY position for the day, not
    # just the offending one.
    BUDGET_CEILING = 580_000

    # UQ Dollar oscillates around a hard $100 anchor (ACF(1) of daily changes
    # = -0.49, half-life 0.7 days). Using the constant rather than a rolling
    # mean costs nothing and leaves zero parameters to fit. Highest capture in
    # the book at 63% and the evenest quarters; do not touch it.
    UQ_ANCHOR = 100.0

    # Boat Party is averaged across all four windows instead of committing to
    # one. Its parameter plateau is badly shaped -- positive at only 8 of 11
    # windows and swinging $100K across the range -- so any single choice is a
    # guess. Averaging makes the choice unnecessary.
    BOAT_WINDOWS = (3, 5, 10, 20)

    # Fintech keeps a single window: its plateau is positive at 11 of 11
    # settings with w=5 sitting mid-plateau, so there is nothing to average
    # away. Ensembling it changes almost nothing ($72,652 vs $75,369 median).
    FINTECH_WINDOW = 5

    # Bread momentum lookback: today's price against a week ago.
    BREAD_LOOKBACK = 7

    # Human-readable description of each signal, keyed by instrument. report.py
    # prints this straight out, so the documented settings can never drift from
    # the ones actually traded. "params" counts values fitted to the data.
    SPEC = {
        "UQ Dollar":         ("no window", "100.0 - P[-1]",                  0),
        "Boat Party Ticket": ("ensemble",  "sign vote, w in 3/5/10/20",      0),
        "Sausage Sizzle":    ("no window", "A + B*dBread + C*dSausage",      3),
        "Fintech Token":     ("w = 5",     "mean(P[-5:]) - P[-1]",           1),
        "Thrifted Jeans":    ("MA 3/15",   "fast-slow, slope>5% over 10d",   4),
        "Bread":             ("k = 7",     "P[-1] - P[-8]  (momentum)",      1),
        "MenuDash":          ("static",    "+1  (always long)",              1),
    }

    # Sausage Sizzle re-prices off the PREVIOUS day's moves in its inputs:
    # d(Sausage) leads d(Sausage Sizzle) by one day at corr +0.61, and
    # d(Bread) at +0.71. Contemporaneous correlation is ~0.02, so this only
    # shows up at lag 1 -- it is a cost passthrough, not a chart pattern.
    # Coefficients are estimated by OLS rather than chosen, which is why it
    # generalises at 90.6% capture. Leave them alone.
    SIZZLE_COEFFS = {"const": 0.0053, "Bread": 0.0769, "Sausage": 1.6780}

    # Thrifted Jeans trades an MA crossover, but only while the slow MA is
    # actually going somewhere. Across the 45-combination grid every setting
    # was positive overall, but the median first half earned -$688: the edge
    # only exists in trending stretches. A 5% slope gate is the most selective
    # of the settings tested and gives the only fit/test ratio near 1.
    TJ_FAST_WINDOW = 3
    TJ_SLOW_WINDOW = 15
    TJ_SLOPE_LOOKBACK = 10     # compare the slow MA against N days ago
    TJ_MIN_SLOPE = 0.05        # need >5% change over that span to trade

    # Fintech volatility filter: go flat after an outsized single-day move,
    # resume once things are calm again.
    #
    # MEASURED, and it earns its place -- but NOT for the reason it was built.
    # The theory was that it would rescue the days 120-150 crash, where a
    # rolling mean straddles two price levels and keeps fading a move that
    # never comes back. It does not: that window loses $16,945 with the filter
    # on or off, and none of the 18 days it pauses fall inside it. What it
    # actually does is sit out three other volatility clusters (days 157-162,
    # 231-237, 325-329), worth +$6,395 on the fitted half and -$3,093 on the
    # held-out half; +$3,302 net, and the book's fit/test gap tightens from
    # -10.8pp to -8.5pp.
    #
    # Treat that as provisional. A change that works for different reasons
    # than its theory predicted is not yet understood, and $3,302 across 18
    # days is thin support for three extra parameters. The days 120-150 loss
    # remains the largest unsolved problem in this instrument.
    #
    # Set to False and compare
    # before trusting it. Thresholds are in units of the instrument's own
    # volatility so they do not need retuning per regime.
    FT_JUMP_PAUSE = True
    FT_JUMP_SIGMA = 3.0        # a move over N rolling std devs starts a pause
    FT_CALM_DAYS = 2           # consecutive quiet days needed to resume
    FT_VOL_WINDOW = 20

    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        self.data = {}  # Historical data of all instruments
        self.positionLimits = {}    # Initialise position limits
        self.day = 0     # Initialise the current day as 0
        self.positions = positions   # Initialise the current positions
        # Jump-filter state. Persists across days because the engine reuses
        # this same object for the whole simulation.
        self.ft_paused = False
        self.ft_calm_run = 0

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
        window contributes only its DIRECTION, not its magnitude.

        Averaging the raw signals instead would quietly reweight the vote: a
        20-day mean sits much further from today's price than a 3-day mean, so
        the longest window would dominate and the ensemble would just be a
        slow signal wearing a disguise. Voting keeps every horizon equal.

        Returns a value in [-1, 1]; 0 when the windows split evenly, which
        leaves the position flat. Windows without enough history yet abstain,
        so the signal starts as soon as the shortest one is usable.
        """
        votes = []
        for window in windows:
            signal = self.reversion_signal(instrument, window)
            if signal is not None:
                votes.append(np.sign(signal))
        if not votes:
            return None
        return float(sum(votes) / len(votes))

    def sizzle_signal(self):
        """
        Predict tomorrow's move in Sausage Sizzle from today's moves in Bread
        and Sausage. Returns None until both inputs have a previous day to
        difference against.
        """
        prediction = self.SIZZLE_COEFFS["const"]
        for name in ("Bread", "Sausage"):
            prices = self.data.get(name)
            if prices is None or len(prices) < 2:
                return None
            prediction += self.SIZZLE_COEFFS[name] * (prices[-1] - prices[-2])
        return prediction

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

    def fintech_is_paused(self):
        """
        Track whether a level shift has put Fintech on the sidelines.

        A move larger than FT_JUMP_SIGMA times its own recent volatility means
        the level has changed, so the rolling mean is now an average of two
        different regimes and cannot be reverted to. Sit out until the price
        has been quiet for FT_CALM_DAYS in a row; any loud day resets that
        count, so a choppy settling period keeps us out until it truly calms.
        """
        prices = self.data["Fintech Token"]
        if len(prices) < self.FT_VOL_WINDOW + 1:
            return False

        recent = np.array(prices[-(self.FT_VOL_WINDOW + 1):], dtype=float)
        moves = np.diff(recent)
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
            return self.UQ_ANCHOR - self.get_current_price(instrument)
        if instrument == "Sausage Sizzle":
            return self.sizzle_signal()
        if instrument == "Thrifted Jeans":
            return self.jeans_signal()
        if instrument == "Boat Party Ticket":
            return self.ensemble_reversion_signal(instrument, self.BOAT_WINDOWS)
        if instrument == "Fintech Token":
            # State has to advance every day, so update the filter before
            # deciding whether to act on it.
            paused = self.fintech_is_paused() if self.FT_JUMP_PAUSE else False
            if paused:
                return 0.0
            return self.reversion_signal(instrument, self.FINTECH_WINDOW)
        if instrument == "Bread":
            prices = self.data[instrument]
            if len(prices) <= self.BREAD_LOOKBACK:
                return None
            return prices[-1] - prices[-(self.BREAD_LOOKBACK + 1)]
        if instrument == "MenuDash":
            return 1.0
        return None

    def enforce_budget(self, desiredPositions):
        """
        Drop whole instruments, cheapest-to-lose first, until the day's total
        exposure fits under BUDGET_CEILING.

        Dropping rather than scaling everything down is deliberate: a
        proportional trim would quietly shrink Sizzle and UQ Dollar -- the two
        most dependable earners -- to make room for whatever is least
        trustworthy. Reverse-priority dropping protects the good signals and
        sacrifices the marginal ones.

        On Round 1 this never fires. It is insurance against Round 2 prices,
        where a level shift in an expensive instrument could push the book over
        the cap and have the engine zero every position for that day.
        """
        def exposure():
            return sum(abs(position) * self.data[instrument][-1]
                       for instrument, position in desiredPositions.items()
                       if position)

        total = exposure()
        if total <= self.BUDGET_CEILING:
            return desiredPositions

        dropped = []
        for instrument in reversed(self.PRIORITY):
            if total <= self.BUDGET_CEILING:
                break
            if desiredPositions.get(instrument):
                desiredPositions[instrument] = 0
                dropped.append(instrument)
                total = exposure()
        print(f"BUDGET GUARD: dropped {', '.join(dropped)} -> ${total:,.0f}")
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

        # IMPLEMENT CODE HERE TO DECIDE WHAT POSITIONS YOU WANT
        #######################################################################
        # Display the current trading day
        print("Starting Algorithm for Day:", self.day)

        # Greedy sizing: take the full position limit in whichever direction
        # the signal points. The four-instrument book peaks around $316K of the
        # $600K daily budget, so there is no need to scale down -- but
        # enforce_budget below is the backstop if Round 2 prices change that.
        for instrument in self.ENABLED:
            signal = self.get_signal(instrument)
            if signal is None or signal == 0:
                # Not enough history yet, or no view today. Stay flat.
                continue
            limit = positionLimits[instrument]
            desiredPositions[instrument] = limit if signal > 0 else -limit
            print(f"{instrument}: ${self.get_current_price(instrument)} "
                  f"signal={signal:+.4f} position={desiredPositions[instrument]}")

        desiredPositions = self.enforce_budget(desiredPositions)

        # Display the end of trading day
        print("Ending Algorithm for Day:", self.day, "\n")
        #######################################################################
        # Return the desired positions
        return desiredPositions
