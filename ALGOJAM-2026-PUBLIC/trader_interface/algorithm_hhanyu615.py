import numpy as np

# Custom trading Algorithm
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

    # UQ Dollar oscillates around a hard $100 anchor (ACF(1) of daily changes
    # = -0.49, half-life 0.7 days). Using the constant rather than a rolling
    # mean costs nothing and leaves zero parameters to fit.
    UQ_ANCHOR = 100.0

    # Mean-reversion lookbacks. Both edges are short-horizon only.
    REVERSION_WINDOWS = {
        "Boat Party Ticket": 5,
        "Fintech Token": 5,
    }

    # Human-readable description of each signal, keyed by instrument. report.py
    # prints this straight out, so the documented settings can never drift from
    # the ones actually traded. "params" counts values fitted to the data.
    SPEC = {
        "UQ Dollar":         ("no window", "100.0 - P[-1]",                   0),
        "Boat Party Ticket": ("w = 5",     "mean(P[-5:]) - P[-1]",            1),
        "Sausage Sizzle":    ("no window", "A + B*dBread + C*dSausage",       3),
        "Fintech Token":     ("w = 5",     "mean(P[-5:]) - P[-1]",            1),
        "Thrifted Jeans":    ("MA 5/20",   "fast-slow, slope>2% over 10d",    4),
        "Sausage":           ("w = 8",     "Donchian breakout, held between",  1),
        "Bread":             ("w = 8",     "Donchian breakout, held between",  1),
        "MenuDash":          ("w = 10",    "-(z-score), |z|>0.5",             2),
    }

    # Sausage Sizzle re-prices off the PREVIOUS day's moves in its inputs:
    # d(Sausage) leads d(Sausage Sizzle) by one day at corr +0.61, and
    # d(Bread) at +0.71. Contemporaneous correlation is ~0.02, so this only
    # shows up at lag 1 -- it is a cost passthrough, not a chart pattern.
    #
    # Coefficients are fixed from a Round 1 fit (R^2 = 0.85). Refitting them
    # on an expanding window instead costs ~$18K, entirely in the early days
    # while the fit warms up; both approaches converge to the same 90.6%
    # capture in the back half, so the fixed version is preferred.
    SIZZLE_COEFFS = {"const": 0.0053, "Bread": 0.0769, "Sausage": 1.6780}

    # Thrifted Jeans trades an MA crossover, but only while the slow MA is
    # actually going somewhere. In a sideways range the slow MA is flat by
    # definition however wildly the fast MA swings, so this filter catches
    # the chop that a gap-size filter misses.
    TJ_FAST_WINDOW = 5
    TJ_SLOW_WINDOW = 20
    TJ_SLOPE_LOOKBACK = 10     # compare the slow MA against N days ago
    TJ_MIN_SLOPE = 0.02        # need >2% change over that span to trade

    # Sausage momentum is short-horizon only. Same-length move autocorrelation
    # peaks at +0.324 for both 5d and 8d -- the highest of any instrument here
    # -- then reverses hard at 20d (-0.354) and 30d (-0.702). Donchian at 8
    # matches that peak. A slope filter was tested and HURT badly ($1,950 vs
    # $10,500): unlike Thrifted Jeans, Sausage has no sustained sideways
    # ranges, so filtering only keeps us out of live trends.
    SAUSAGE_WINDOW = 8

    # Bread momentum peaks at a 7-day horizon (+0.203) and reverses hard by
    # 20d (-0.407) and 30d (-0.662) -- same shape as Sausage. 
    
    BREAD_WINDOW = 8

    # MenuDash posts a noisy read on a smooth underlying cost. The noise is
    # SMALL and FREQUENT (1-3 day wobbles), not large and rare
    MENUDASH_WINDOW = 10
    MENUDASH_THRESHOLD = 0.5


    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        self.data = {}  # Historical data of all instruments
        self.positionLimits = {}    # Initialise position limits
        self.day = 0     # Initialise the current day as 0
        self.positions = positions   # Initialise the current positions
        # Donchian holds its direction between breakouts, so it needs state
        # that survives across days. The engine reuses this object all year.
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

    def sausage_signal(self):
        """
        Donchian breakout: a new 8-day high flips us long, a new 8-day low
        flips us short. Between the two extremes we HOLD the last direction
        rather than re-deciding, which is what stops it churning on noise.
        Returns 0 until the first breakout, i.e. flat.
        """
        prices = self.data["Sausage"]
        window = self.SAUSAGE_WINDOW
        if len(prices) < window:
            return None

        recent = np.array(prices[-window:], dtype=float)
        if prices[-1] >= recent.max():
            self.sausage_direction = 1
        elif prices[-1] <= recent.min():
            self.sausage_direction = -1
        return float(self.sausage_direction)

    def bread_signal(self):
        """
        Donchian breakout on Bread. Same mechanic as sausage_signal: a new
        8-day high flips us long, a new 8-day low flips us short, and between
        the two we HOLD the last direction rather than re-deciding.
        Returns 0 until the first breakout, i.e. flat.
        """
        prices = self.data["Bread"]
        window = self.BREAD_WINDOW
        if len(prices) < window:
            return None

        recent = np.array(prices[-window:], dtype=float)
        if prices[-1] >= recent.max():
            self.bread_direction = 1
        elif prices[-1] <= recent.min():
            self.bread_direction = -1
        return float(self.bread_direction)

    def menudash_signal(self):
        """
        Positive when MenuDash sits below its own recent mean, i.e. when we
        expect the posted price to drift back up. Flat when the deviation is
        too small to be worth trading. Returns None until there is enough
        history.
        """
        prices = self.data["MenuDash"]
        window = self.MENUDASH_WINDOW
        if len(prices) < window:
            return None

        recent = np.array(prices[-window:], dtype=float)
        sd = recent.std()
        if sd == 0:
            return None

        z = (recent[-1] - recent.mean()) / sd
        if abs(z) < self.MENUDASH_THRESHOLD:
            return 0.0
        # Above the mean -> expect a fall -> negative signal -> short
        return -z

    def get_signal(self, instrument):
        """Signal for one instrument; sign is the direction we want to hold."""
        if instrument == "UQ Dollar":
            return self.UQ_ANCHOR - self.get_current_price(instrument)
        if instrument == "Sausage Sizzle":
            return self.sizzle_signal()
        if instrument == "Thrifted Jeans":
            return self.jeans_signal()
        if instrument == "Sausage":
            return self.sausage_signal()
        if instrument == "Bread":
            return self.bread_signal()
        if instrument == "MenuDash":
            return self.menudash_signal()
        if instrument in self.REVERSION_WINDOWS:
            return self.reversion_signal(instrument, self.REVERSION_WINDOWS[instrument])
        return None

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
        # the signal points. The enabled book peaks around $290K of the
        # $600K daily budget, so there is no need to scale positions down.
        for instrument in self.ENABLED:
            signal = self.get_signal(instrument)
            if signal is None or signal == 0:
                # Not enough history yet, or no view today. Stay flat.
                continue
            limit = positionLimits[instrument]
            desiredPositions[instrument] = limit if signal > 0 else -limit
            print(f"{instrument}: ${self.get_current_price(instrument)} "
                  f"signal={signal:+.4f} position={desiredPositions[instrument]}")

        # Display the end of trading day
        print("Ending Algorithm for Day:", self.day, "\n")
        #######################################################################
        # Return the desired positions
        return desiredPositions