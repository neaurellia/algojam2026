import numpy as np

class Algorithm():

    def __init__(self, positions):
        self.data = {}              # Historical price data for each instrument
        self.positionLimits = {}    # Position limits for each instrument
        self.day = 0
        self.positions = positions

        # --- Fintech Token jump-pause state ---------------------------------
        # These persist across days because the engine reuses this same
        # Algorithm object for the entire simulation.
        self.ft_paused = False    # are we sitting out after a jump?
        self.ft_calm_run = 0      # consecutive calm days seen while paused

        # Fintech Token strategy parameters
        self.ft_window = 15            # days used for mean/std
        self.ft_z_entry = 0.5          # trade once price is this many std out
        self.ft_jump_threshold = 0.05  # a >5% one-day move counts as a jump
        self.ft_stable_days = 2        # calm days in a row needed to resume
        self.ft_stable_threshold = 0.025   # a "calm" day moves less than 2.5%

        # --- Thrifted Jeans parameters ---------------------------------------
        self.tj_fast_window = 5
        self.tj_slow_window = 20

        # Trend filter: only trade when the SLOW MA is actually going somewhere.
        # In a sideways range the slow MA is flat by definition, however wildly
        # the fast MA swings around it - so this catches the chop that a
        # gap-size filter misses. 
        self.tj_slope_lookback = 10    # compare slow MA against N days ago
        self.tj_min_slope = 0.02       # need >2% change over that span

        # --- Sausage Sizzle parameters ----------------------------------------
        # Sizzle is a derived instrument: its price is its ingredients added up,
        # booked with a one-day lag ("today's price comes off yesterday's
        # shopping"). Fitted on Round 1:
        #     dSizzle_t = 0.0053 + 0.0769*dBread_(t-1) + 1.6780*dSausage_(t-1)
        #     R^2 = 0.8511, directional accuracy 82.6%
        
        # MenuDash is deliberately excluded. 
        self.ss_const = 0.0053
        self.ss_bread_coef = 0.0769
        self.ss_sausage_coef = 1.6780

        # Deadband: skip predictions smaller than this. TESTED - every non-zero
        # value REDUCED profit (0.05 cost ~$11k), because even small-magnitude
        # predictions here are still directionally correct. Leave at 0.0.
        self.ss_min_prediction = 0.0

        # Fraction of the position limit to use. Drop below 1.0 if you need
        # budget headroom - Sizzle at full limit is roughly $114k of exposure.
        self.ss_position_fraction = 1.0

    def get_current_price(self, instrument):
        return self.data[instrument][-1]

    ###########################################################################
    # ------------------------- HELPER FUNCTIONS ------------------------------
    ###########################################################################

    def rolling_mean(self, arr, window):
        """Simple rolling mean."""
        if len(arr) < window:
            return None
        return np.mean(arr[-window:])

    def rolling_std(self, arr, window):
        """Simple rolling standard deviation."""
        if len(arr) < window:
            return None
        return np.std(arr[-window:])

    def z_score(self, arr, window):
        """Compute z-score for mean reversion."""
        mu = self.rolling_mean(arr, window)
        sigma = self.rolling_std(arr, window)
        if mu is None or sigma is None or sigma == 0:
            return 0
        return (arr[-1] - mu) / sigma

    def rate_of_change(self, arr, period):
        """Rate of change over N days."""
        if len(arr) < period + 1:
            return 0
        return (arr[-1] - arr[-period]) / arr[-period]

    def daily_move(self, arr):
        """
        Size of today's move as a fraction of yesterday's price.
        Returns None when there's no previous day to compare against.
        """
        if len(arr) < 2 or arr[-2] == 0:
            return None
        return abs(arr[-1] - arr[-2]) / arr[-2]

    def update_ft_jump_state(self, prices):
        """
        Decide whether the Fintech Token strategy should be sitting out today.

        Why this exists: when the token jumps to a new level, the lookback
        window straddles TWO different price levels, so its "mean" is a price
        the token never actually trades at. Mean reversion would then bet
        confidently on a snap-back to the OLD level that never comes.

        Logic:
          - A move bigger than ft_jump_threshold means a jump just happened:
            pause immediately and reset the calm counter.
          - While paused, count consecutive calm days. After ft_stable_days
            of them, the price has settled at its new level, so resume.
          - Any non-calm day while paused resets the counter to zero, so a
            choppy settling period keeps us out until it genuinely quietens.
        """
        move = self.daily_move(prices)

        if move is None:
            return

        if move > self.ft_jump_threshold:
            self.ft_paused = True
            self.ft_calm_run = 0
            print(f"Fintech Token: [JUMP DETECTED] {move*100:.2f}% move - pausing")

        elif self.ft_paused:
            if move < self.ft_stable_threshold:
                self.ft_calm_run += 1
                if self.ft_calm_run >= self.ft_stable_days:
                    self.ft_paused = False
                    self.ft_calm_run = 0
                    print("Fintech Token: [STABILISED] resuming mean reversion")
            else:
                self.ft_calm_run = 0

    def tj_slow_ma_slope(self, prices):
        """
        Fractional change in the slow moving average over the last
        tj_slope_lookback days. Near zero = flat = sideways market.
        Returns None if there isn't enough history.
        """
        s = self.tj_slow_window
        lb = self.tj_slope_lookback

        # Need enough data for the slow MA both today and lb days ago
        if len(prices) < s + lb:
            return None

        ma_now = np.mean(prices[-s:])
        ma_then = np.mean(prices[-(s + lb):-lb])

        if ma_then == 0:
            return None

        return (ma_now - ma_then) / ma_then

    def ss_predict_next_move(self):
        """
        Predict Sausage Sizzle's NEXT-day price change from today's changes in
        Bread and Sausage.

        Returns None if any component is missing or too short.
        """
        needed = ["Bread", "Sausage"]
        for ins in needed:
            if ins not in self.data or len(self.data[ins]) < 2:
                return None

        bread = self.data["Bread"]
        sausage = self.data["Sausage"]

        # Today's change in each ingredient
        d_bread = bread[-1] - bread[-2]
        d_sausage = sausage[-1] - sausage[-2]

        return (self.ss_const
                + self.ss_bread_coef * d_bread
                + self.ss_sausage_coef * d_sausage)

    ###########################################################################
    # ------------------------- MAIN TRADING LOGIC ----------------------------
    ###########################################################################

    def get_positions(self):

        desiredPositions = {ins: 0 for ins in self.positionLimits}

        print("Starting Algorithm for Day:", self.day)

        #######################################################################
        # ------------------------- UQ DOLLAR STRATEGY -------------------------
        #######################################################################

        uq = "UQ Dollar"
        if uq in self.data and len(self.data[uq]) > 40:

            prices = self.data[uq]

            # Mean reversion parameters
            window = 30
            z = self.z_score(prices, window)

            print(f"UQ Dollar price: {prices[-1]:.2f}, z-score={z:.3f}")

            # Thresholds for mean reversion
            upper = 0.3
            lower = -0.3

            if z > upper:
                # Price too high → short
                desiredPositions[uq] = -self.positionLimits[uq]
            elif z < lower:
                # Price too low → long
                desiredPositions[uq] = self.positionLimits[uq]
            else:
                desiredPositions[uq] = 0

        
        bpt = "Boat Party Ticket"

        if bpt in self.data:
            # position limit
            limit = self.positionLimits[bpt]   

            day = self.day % 365   # wrap around yearly cycle

            # --- PEAK WINDOWS (short) ---
            peak_windows = [
            (33,63),    # Semester 1 crest
            (112, 118),   # post sem1 midsem exams
            (190, 205),  #Semester 2 crest
            (248, 258) #Sem 2 post midsem exams
            ]

            # --- DIP WINDOWS (long) ---
            dip_windows = [
            (0, 10),   # pre sem1 trough
            (90, 100),  # sem1 midsem exams
            (155, 167),   # sem 1 exams probably
            (230, 237),  #sem2 midsem exams
            (296,310) #sem2 finals
            ]

            # --- EOY window (stable) (mean reversion)
            eoy_windows = [
            (320, 364)
            ]

            # Check if current day is in any peak window
            in_peak = any(start <= day <= end for start, end in peak_windows)
            in_dip  = any(start <= day <= end for start, end in dip_windows)
            in_eoy = any(start <= day <= end for start, end in eoy_windows)

            if in_peak:
                desiredPositions[bpt] = -limit   # short the peak
                print(f"Boat Party Ticket: SHORT (peak window)")

            elif in_dip:
                desiredPositions[bpt] = limit    # long the dip
                print(f"Boat Party Ticket: LONG (dip window)")

            elif in_eoy:
                prices = self.data[bpt]
                window = 20

                if len(prices) >= window:
                    mu_bpt = np.mean(prices[-window:])
                    sigma_bpt = np.std(prices[-window:])

                    if sigma_bpt ==0:
                        sigma_bpt = 1e-6

                    z = (prices[-1] - mu_bpt) / sigma_bpt

                    # mean reversion thresholds
                    upper = 0.5
                    lower = -0.5

                    if z > upper:
                        desiredPositions[bpt] = -limit   # price too high → short
                        print("Boat Party Ticket: SHORT (late-year mean reversion)")
                    elif z < lower:
                        desiredPositions[bpt] = limit    # price too low → long
                        print("Boat Party Ticket: LONG (late-year mean reversion)")
                    else:
                        desiredPositions[bpt] = 0        # neutral
                        print("Boat Party Ticket: FLAT (late-year mean reversion)")

            else:
                desiredPositions[bpt] = 0        # neutral outside seasonal zones
                print(f"Boat Party Ticket: FLAT (neutral window)")

        #######################################################################
        # ---------------------- FINTECH TOKEN STRATEGY ------------------------
        #######################################################################

        # Strategy: mean reversion, paused around jumps.
        #
        # The token sits flat for weeks, oscillating around a level, then
        # jumps to a new level and stays there. The oscillation is what we
        # harvest; the jumps are what we avoid.
        #
        # NOTE: momentum / MA-crossover was tested here and LOST money.
        # Daily returns are negatively autocorrelated (-0.135), i.e.
        # mean-reverting, so trend-following bets the wrong way despite the
        # trending *look* of the chart.

        ft = "Fintech Token"
        if ft in self.data:

            prices = self.data[ft]
            limit = self.positionLimits[ft]

            # Step 1: update whether a jump has put us on the sidelines
            self.update_ft_jump_state(prices)

            # Step 2: only trade if we're not sitting out
            if self.ft_paused:
                desiredPositions[ft] = 0
            else:
                z = self.z_score(prices, self.ft_window)

                # z_score() returns 0 while warming up, which correctly
                # leaves us flat until there's enough history
                if z > self.ft_z_entry:
                    desiredPositions[ft] = -limit   # overpriced → short
                elif z < -self.ft_z_entry:
                    desiredPositions[ft] = limit    # underpriced → long
                else:
                    desiredPositions[ft] = 0        # inside noise band → flat

            state = "PAUSED" if self.ft_paused else "active"
            z_now = self.z_score(prices, self.ft_window)
            print(f"Fintech Token price: {prices[-1]:.2f}, z={z_now:+.3f}, "
                  f"[{state}] -> position {desiredPositions[ft]}")

        #######################################################################
        # --------------------- THRIFTED JEANS STRATEGY ------------------------
        #######################################################################
        # MA crossover, but only when the market is actually trending.
        # Days 50-125 of Round 1 is a sideways range where MA(5,20) flipped
        # direction 11 times in 75 days and lost $33,552. The slope filter
        # keeps us flat through that kind of chop.

        tj = "Thrifted Jeans"
        if tj in self.data and tj in self.positionLimits:

            prices = self.data[tj]
            limit = self.positionLimits[tj]

            fast = self.rolling_mean(prices, self.tj_fast_window)
            slow = self.rolling_mean(prices, self.tj_slow_window)
            slope = self.tj_slow_ma_slope(prices)

            if fast is None or slow is None or slope is None:
                # Still warming up
                desiredPositions[tj] = 0
                print(f"Thrifted Jeans: warming up -> position 0")
            elif abs(slope) < self.tj_min_slope:
                # Slow MA is flat - sideways market, don't get whipsawed
                desiredPositions[tj] = 0
                print(f"Thrifted Jeans: {prices[-1]:.2f}, slope={slope:+.3f} "
                      f"-> FLAT (no trend)")
            else:
                direction = 1 if fast > slow else -1
                position = int(direction * limit)
                desiredPositions[tj] = max(-limit, min(limit, position))
                print(f"Thrifted Jeans: {prices[-1]:.2f}, slope={slope:+.3f}, "
                      f"dir={direction:+d} -> position {desiredPositions[tj]}/{limit}")

        #######################################################################
        # --------------------- SAUSAGE SIZZLE STRATEGY ------------------------
        #######################################################################
        # Not a chart-pattern strategy - this reconstructs the instrument's
        # actual pricing formula. Predict tomorrow's move from today's
        # ingredient moves, then take the corresponding side at full size.

        ss = "Sausage Sizzle"
        if ss in self.data and ss in self.positionLimits:

            limit = self.positionLimits[ss]
            predicted_move = self.ss_predict_next_move()

            if predicted_move is None:
                # Missing ingredient data (or day 0) - stay flat
                desiredPositions[ss] = 0
                print("Sausage Sizzle: no ingredient data -> position 0")

            elif abs(predicted_move) < self.ss_min_prediction:
                # Below deadband (disabled by default - see note in __init__)
                desiredPositions[ss] = 0
                print(f"Sausage Sizzle: pred={predicted_move:+.4f} -> FLAT (deadband)")

            else:
                # Predicted up -> long, predicted down -> short
                direction = 1 if predicted_move > 0 else -1
                position = int(direction * limit * self.ss_position_fraction)
                desiredPositions[ss] = max(-limit, min(limit, position))

                print(f"Sausage Sizzle: {self.data[ss][-1]:.2f}, "
                      f"pred_move={predicted_move:+.4f} "
                      f"-> position {desiredPositions[ss]}/{limit}")


                        
        #######################################################################
        print("Ending Algorithm for Day:", self.day, "\n")
        return desiredPositions