import numpy as np
from scipy.special import comb, factorial
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from copy import deepcopy
from collections import Counter

import time


class Balatro:

    NUM_RANKS = 13
    NUM_SUITS = 4
    RANKS = np.array(["K", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q"])
    SUITS = np.array(["C", "H", "S", "D"])

    def __init__(self, num_simulations, num_decks=1, strategy=None, seed=None, verbose=False):
        # init
        self.num_simulations = num_simulations
        self.num_decks = num_decks
        self.strategy = strategy # How cards are chosen
        self.seed = seed
        self.verbose = verbose

        if self.seed and self.verbose:
            print(f"Using seed={self.seed}")
        self.rng = np.random.default_rng(seed=self.seed)

        # Deck setup
        self.deck = np.arange(52*self.num_decks) % 52
        self.deck_arr = np.asarray(self.deck)
        self.seen_cards = np.zeros((self.num_simulations, 52))
        self.rank_arr = self.deck_arr % self.NUM_RANKS
        self.suit_arr = self.deck_arr % self.NUM_SUITS

        # Default Balatro rules
        self.max_hand_size = 8
        self.max_num_hands = 4
        self.max_num_discards = 3
        self.num_hands_discards_left = np.tile(np.array([self.max_num_hands, self.max_num_discards]), (self.num_simulations,1))
        self.gold = 3
        self.chips_mult = { # (chip, mult)
            "Flush Five": [160, 16],
            "Flush House": [140, 14],
            "Five of a Kind": [120, 12],
            "Straight Flush": [100, 8],
            "Four of a Kind": [60, 7],
            "Full House": [40, 4],
            "Flush": [35, 4],
            "Straight": [30, 4],
            "Three of a Kind": [30, 3],
            "Two Pair": [20, 2],
            "Pair": [10, 2],
            "High Card": [5, 1],
        }

        # Poker References
        self.poker_hand_size = 5
        self.poker_hands = np.array(list(self.chips_mult.keys()))
        self.straight_references = np.array(
            [np.arange(1+i,i+6) for i in range(8)] + 
            [[9, 10, 11, 12, 0],[10, 11, 12, 0, 1]])

        # Matrix helpers
        self.row_selector = np.arange(self.num_simulations)[:,None] # help select all rows. Shape: (num_simulations, 1)
    
    def get_shuffled_deck(self):
        """
        Makes a matrix of shuffled decks per row

        Returns:
            shuffled_deck (np.array): Array of shuffled cards. Shape: (num_simulations, shuffled cards in self.deck)
        """
        shuffled_deck = np.array([self.rng.permutation(self.deck_arr) for _ in range(self.num_simulations)]) # Shape: (num_simulations, cards in deck)
        return shuffled_deck

    def simulate_one_round(self):
        """
        Simulates one round of Balatro
        """

        shuffled_deck = self.get_shuffled_deck()
        current_hand_idx = np.tile(np.arange(self.max_hand_size), (self.num_simulations,1)) # ith row says which card index to grab per simulation. Shape: (num_simulations, self.max_hand_size)
        current_hand = shuffled_deck[self.row_selector, current_hand_idx]

        total_score = np.zeros(self.num_simulations)
        is_done = np.repeat(False, self.num_simulations)
        ith_hand_played = 0
        while ~np.all(is_done):
            if self.verbose:
                print(f"Hand {ith_hand_played}")
                print(self.print_hand(current_hand, sort=True))
            # 1. evaluate hand whether to play hand or discard
            game_state = self.hand_eval(current_hand, shuffled_deck, current_hand_idx)

            # 2. score or draw cards to restore hand
            total_score += game_state["score"]

            # Update for next loop
            self.num_hands_discards_left[:, 0] -= game_state["play_hand"].astype(int) # hands
            self.num_hands_discards_left[:, 1] -= ~game_state["play_hand"].astype(int) # discards
            is_done |= (self.num_hands_discards_left[:, 0] <= 0)
            ith_hand_played += 1

        return total_score, game_state["target_poker_hand"]
    
    def print_hand(self, hand_matrix, sort=False):
        """
        Integer encoding to human readable str

        Args:
            hand_matrix (np.array): Array of int. Size: (num_simulations, self.max_hand_size)
        Returns:
            hands_str (np.array): Array of str. Size: (num_simulations, self.max_hand_size)
        """
        idx_of_ranks_in_hand = hand_matrix % self.NUM_RANKS
        ranks_in_hand = self.RANKS[idx_of_ranks_in_hand]

        idx_of_suits_in_hand = hand_matrix % self.NUM_SUITS
        suits_in_hand = self.SUITS[idx_of_suits_in_hand]
        
        hands_str = np.char.add(ranks_in_hand, suits_in_hand)
        if sort:
            hands_str = np.sort(hands_str)

        return hands_str

    def hand_eval(self, current_hand, shuffled_deck, current_hand_idx):
        """
        Evaluate the hand to get the target poker hand and cards to keep

        Args:
            current_hand (np.array): Array of cards in hand. Shape: (num_simulations, self.max_hand_size)
        Returns:
            game_state (dict): Dict of current game state. See return of self.decision.

        """


        game_state = self.decision(current_hand)
        return game_state


    def get_score(self, hand, target_poker_hand):
        """
        Scores chosen hand and target_poker_hand

        Args:
            hand (np.array): Array of cards in hand. Shape: (num_simulations, self.poker_hand_size)
            target_poker_hand (np.array): Array of poker hand str. Shape: (num_simulations,)
        returns:
            score_arr (np.array): Array of int score. Shape: (num_simulations,)
        """
        # TODO: Add processing to only use the 5 cards of the poker hand
        # if len(hand) > self.poker_hand_size:
        #     raise ValueError(f"Received invalid number of cards. Expected at most {self.poker_hand_size} cards to score a hand.")
        hand_ranks = hand % self.NUM_RANKS
        hand_ranks_face_corrected = np.where(np.logical_or(hand % self.NUM_RANKS == 0, hand % self.NUM_RANKS > 10), 10 ,hand % self.NUM_RANKS) # J,Q,K are valued as 10
        hand_ranks_ace_corrected = np.where(hand_ranks_face_corrected == 1, 11, hand_ranks_face_corrected) # A are valued as 11
        hand_rank_sum = hand_ranks.sum(axis=1)
        chips, mult = np.array([self.chips_mult[poker_hand] for poker_hand in target_poker_hand]).T
        score_arr = (hand_rank_sum + chips) * mult
        return score_arr

    def decision(self, current_hand):
        """
        Function that chooses how to handle cards

        Args:
            current_hand (np.array): Array of cards in hand. Shape: (num_simulations, self.max_hand_size)
            current_poker_hand_matches (np.array): Array of bool indicating whether poker hand is available. Shape: (num_simulations, len(self.poker_hands))
        Returns:
            game_state (dict): Dict of current gamestate. 
                - play_hand (np.array): Bool array if True, hand will play best poker hand available to be score. Otherwise, do not score. Shape: (num_simulations,)
                - score (np.array): Array of int score. Shape: (num_simulations,)
                - cards_kept (list): list of cards kept in hand per simulation. len = num_simulations
                - target_poker_hand (np.array): Array of poker hand str deemed as the best target to aim. Shape: (num_simulations,)
                - current_poker_hand_matches (np.array): Array of bool of current poker hand available. Shape: (num_simulations, len(self.poker_hands))
                - poker_hand_prob (np.array): Array of probabilities the next hand will be a given poker hand. Shape: (num_simulations, len(self.poker_hands))
        """
        strategy = self.strategy if self.strategy else "one_hand"
        valid_strategies = ["one_hand", "random", "greedy", "likelihood", "breadth", "depth", "neural_network"]
        game_state = {}

        current_poker_hand_matches, poker_hands_arr = self.get_poker_hand(current_hand=current_hand)
        
        if strategy == "one_hand":
            # Play only one hand to be scored.
            self.num_hands_discards_left[:, 0] = 0 # zeroing out hands to stop after this
            play_hand = np.repeat(True, self.num_simulations)
            cards_kept = current_hand
            target_poker_hand = self.poker_hands[current_poker_hand_matches.argmax(axis=1)]
            poker_hand_prob = current_poker_hand_matches.astype(float)
            
        elif strategy == "random":
            # No evaluation, randomly pick cards to discard
            play_hand = np.repeat(False, self.num_simulations)
            num_cards_to_keep = self.poker_hand_size
            cards_to_keep_idx = np.argpartition(self.rng.random(current_hand.shape), kth=num_cards_to_keep-1, axis=1)[:, :num_cards_to_keep]
            cards_kept = current_hand[self.row_selector, cards_to_keep_idx]
            target_poker_hand = self.poker_hands[current_poker_hand_matches.argmax(axis=1)]
            poker_hand_prob = np.zeros((self.num_simulations, len(self.poker_hands)))

        elif strategy == "greedy":
            # Evaluate which cards to keep based on greedy metrics
            # Keep cards for most complete poker hands
            pass

        elif strategy == "likelihood":
            # Evaluate which cards to keep based on the best expected score
            # This is gonna be hard to implement
            pass
        
        elif strategy == "breadth":
            # Evaluate by beadth first search
            #TODO: Think more about how this can be done
            pass

        elif strategy == "depth":
            # Evaluate by depth first search
            #TODO: Think more about how this can be done
            pass

        elif strategy == "neural_network":
            # Evaluation based on decisions from a neural network
            #TODO: Make an RL neural network
            pass

        else:
            raise ValueError(f"Unrecognized strategy. Expected {valid_strategies} but received strategy")

        score = self.get_score(current_hand, target_poker_hand) * play_hand.astype(int) # zero out score of unplayed hands

        game_state = {
            "play_hand": play_hand,
            "score": score,
            "cards_kept": cards_kept,
            "target_poker_hand": target_poker_hand,
            "current_poker_hand_matches": current_poker_hand_matches,
            "poker_hand_prob": poker_hand_prob,
        }
        return game_state

    def get_count(self, arr, n):
        """
        Gets the the suit or rank count per row 

        Args:
            arr (np.array): 2D Array to get the count. Shape: (num_simulations, n)
            n (int): modulo number. 13 for rank, 4 for suit
        Returns:
            count_arr (np.array):
        """
        count_arr = ((arr[:, :, None] % n) == np.arange(n)).sum(axis=1)
        return count_arr
    
    def get_rank_count(self, arr):
        """
        Gets the the rank count per row 

        Args:
            arr (np.array): 2D Array to get the count. Shape: (num_simulations, 13)
        Returns:
            rank_count (np.array):
        """
        rank_count = self.get_count(arr, self.NUM_RANKS)
        return rank_count

    def get_suit_count(self, arr):
        """
        Gets the the suit count per row 

        Args:
            arr (np.array): 2D Array to get the count. Shape: (num_simulations, 4)
        Returns:
            suit_count (np.array):
        """
        suit_count = self.get_count(arr, self.NUM_SUITS)
        return suit_count

    def get_poker_hand(self, current_hand):
        """
        Identify possible poker hands in current hadn and which cards are involved

        Args:

        Returns:
            current_poker_hand_matches (np.array): Bool of array matching poker hands. Shape: (num_simulations, len(self.poker_hands))
            poker_hands_arr (np.array): Int array representing poker hands. Shape: (num_simulations, self.poker_hand_size)
        """
        current_hand = np.array([
            [0,2,10,10,10,10,10,6], # Flush Five
            [0,2,20,20,20,30,30,6], # Flush House
            [0,2,40,40,40,1,1,6],
            [0,2,0,0,13,26,39,6] # 5 of a kind
        ])

        # Slicing [:, :, None] compares each element with the arange
        rank_count = self.get_rank_count(current_hand) # shape: (num_simulations, NUM_RANKS)
        suit_count = self.get_suit_count(current_hand) # shape: (num_simulations, NUM_SUITS)
        card_count = self.get_count(current_hand, 52) # Shape: (num_simulations, 52)
        self.seen_cards += card_count


        poker_hand_args = {
            "current_hand": current_hand,
            "rank_count": rank_count,
            "suit_count": suit_count,
            "card_count": card_count,
        }
        is_flush_five, flush_five_poker_hand = self.has_flush_five(**poker_hand_args)
        is_flush_house, flush_house_poker_hand = self.has_flush_house(**poker_hand_args)
        is_five_of_a_kind, five_of_a_kind_poker_hand = self.has_5_of_a_kind(**poker_hand_args)
        is_straight_flush, straight_flush_poker_hand = self.has_straight_flush(**poker_hand_args)
        is_four_of_a_kind, four_of_a_kind_poker_hand = self.has_4_of_a_kind(**poker_hand_args)
        is_full_house, full_house_poker_hand = self.has_full_house(**poker_hand_args)
        is_flush, flush_poker_hand = self.has_flush(**poker_hand_args)
        is_straight, straight_poker_hand = self.has_straight(**poker_hand_args)
        is_three_of_a_kind, three_of_a_kind_poker_hand = self.has_3_of_a_kind(**poker_hand_args)
        is_two_pair, two_pair_poker_hand = self.has_two_pair(**poker_hand_args)
        is_pair, pair_poker_hand = self.has_2_of_a_kind(**poker_hand_args)
        is_high_card, high_card_poker_hand = self.has_high_card(**poker_hand_args)
        current_poker_hand_matches = np.vstack([
            is_flush_five,
            is_flush_house,
            is_five_of_a_kind,
            is_straight_flush,
            is_four_of_a_kind, 
            is_full_house, 
            is_flush,
            is_straight, 
            is_three_of_a_kind,
            is_two_pair, 
            is_pair,
            is_high_card,
            ]).T # Shape: (num_simulations, len(self.poker_hands))

        poker_hands_arr = np.sum([
            flush_five_poker_hand, 
            flush_house_poker_hand, 
            five_of_a_kind_poker_hand, 
            straight_flush_poker_hand, 
            four_of_a_kind_poker_hand,
            full_house_poker_hand, 
            flush_poker_hand, 
            straight_poker_hand, 
            three_of_a_kind_poker_hand, 
            high_card_poker_hand,
            pair_poker_hand,
            high_card_poker_hand,
            ],axis=1)
        return current_poker_hand_matches, poker_hands_arr


    def has_n_of_a_kind(self, current_hand, rank_count, suit_count, card_count, n):
        """
        Checks if array of hands each has a n-of-a-kind hand

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
            n (int): num of ranks to match
        Returns:
            is_n_of_a_kind (np.array): Bool array if each simulation is a n-of-a-kind. Shape: (num_simulations,)
        """
        if n < 1:
            raise ValueError(f"n must be a positive integer. Received {n}")
        at_least_n_count = rank_count >= n
        n_of_a_kind_mask = at_least_n_count.any(axis=1)
        identified_rank = np.argmax(at_least_n_count, axis=1)
        card_matches_n_of_a_kind = current_hand % 13 == identified_rank[:,None]
        col_of_cards_to_keep = np.argsort(~card_matches_n_of_a_kind, axis=1)[:,:n] # Shape: (num_simulations, n)
        n_of_a_kind_poker_hand = current_hand[self.row_selector, col_of_cards_to_keep] * n_of_a_kind_mask[:,None]
        #np.where(np.take_along_axis(card_matches_n_of_a_kind, col_of_cards_to_keep, axis=1), n_of_a_kind_poker_hand, 0)
        is_n_of_a_kind = n_of_a_kind_mask
        return is_n_of_a_kind, n_of_a_kind_poker_hand

    def has_flush_five(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a flush five

        Args:
            card_count (np.array): Array of card counts. Shape: (num_simulations, 52)
        Returns:
            is_flush_five (np.array): Bool array if each simulation is a flush five. Shape: (num_simulations,)
            flush_five_poker_hand (np.array): Int array of cards contributing to the flush five. Shape: (num_simulations, self.poker_hand_size)
        """
        is_5_duplicates = card_count == 5
        is_flush_five = (is_5_duplicates).any(axis=1)
        flush_five_card = self.deck[np.argmax(is_5_duplicates, axis=1)]
        flush_five_poker_hand = np.tile(flush_five_card[:,None], (1, self.poker_hand_size))
        return is_flush_five, flush_five_poker_hand
        
    def has_flush_house(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a flush house

        Args:
            card_count (np.array): Array of card counts. Shape: (num_simulations, 52)
        Returns:
            is_flush_house (np.array): Bool array if each simulation is a flush house. Shape: (num_simulations,)
            flush_house_poker_hand (np.array): Int array of cards contributing to flush house. shape: (num_simulations, self.poker_hand_size)
        """
        is_triple = card_count == 3
        is_double = card_count == 2

        is_triple_flush = (is_triple).any(axis=1)
        is_two_rank_duplicates = (is_triple | is_double).sum(axis=1) >= 2 # has two ranks each with duplicates

        # Pick out relevant cards
        triple_duplicates = np.argwhere(is_triple) 
        double_duplicates = np.argwhere(is_double)
        flush_house_poker_hand = np.zeros((self.num_simulations,self.poker_hand_size))
        flush_house_poker_hand[triple_duplicates[:,0], :3] = triple_duplicates[:,1][:,None]
        flush_house_poker_hand[double_duplicates[:,0], 3:] = double_duplicates[:,1][:,None]

        is_flush_house = is_triple_flush & is_two_rank_duplicates
        flush_house_poker_hand = flush_house_poker_hand * is_flush_house[:,None] # zero out non relevant hands
        return is_flush_house, flush_house_poker_hand

    def has_5_of_a_kind(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a 5-of-a-kind

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_5_of_a_kind (np.array): Bool array if each simulation is a 5-of-a-kind. Shape: (num_simulations,)
            five_of_a_kind_poker_hand (np.array): Int array of cards contributing to 5 of a kind. Shape: (num_simulations, self.poker_hand_size)
        """
        is_5_of_a_kind, five_of_a_kind_poker_hand = self.has_n_of_a_kind(current_hand, rank_count, suit_count, card_count, n=5)
        return is_5_of_a_kind, five_of_a_kind_poker_hand

    def has_straight_flush(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a straight flush

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
            suit_count (np.array): Array of suit counts. Shape: (num_simulations, NUM_SUITS)
        Returns:
            is_straight_flush (np.array): Bool array if each simulation is a high card. Shape: (num_simulations,)
        """
        is_flush, a = self.has_flush(current_hand, rank_count, suit_count, card_count)
        is_straight, b = self.has_straight(current_hand, rank_count, suit_count, card_count)
        is_straight_flush = is_flush & is_straight
        return is_straight_flush

    def has_4_of_a_kind(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a 4-of-a-kind

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_4_of_a_kind (np.array): Bool array if each simulation is a 4-of-a-kind. Shape: (num_simulations,)
            four_of_a_kind_poker_hand (np.array): Int array of cards contributing to a 4 of a kind. Shape: (num_simulations, self.poker_hand_size)
        """
        is_4_of_a_kind, four_of_a_kind_poker_hand = self.has_n_of_a_kind(current_hand, rank_count, suit_count, card_count, n=4)
        return is_4_of_a_kind, four_of_a_kind_poker_hand

    def has_full_house(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a full house
        This relies on two pair returning true as a subset of a full house

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_full_house (np.array): Bool array if each simulation is a full house. Shape: (num_simulations,)
        """
        is_3_of_a_kind = self.has_3_of_a_kind(rank_count=rank_count)
        is_two_pair = self.has_two_pair(rank_count=rank_count)
        is_full_house = is_3_of_a_kind & is_two_pair
        return is_full_house

    def has_flush(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a flush

        Args:
            suit_count (np.array): Array of suit counts. Shape: (num_simulations, NUM_SUITS)
        Returns:
            is_flush (np.array): Bool array if each simulation is a flush. Shape: (num_simulations,)
            flush_poker_hand (np.array): Int array of cards contributing to a flush. Shape: (num_simulations, self.poker_hand_size)
        """
        flush_suit = suit_count >= self.poker_hand_size
        is_flush = flush_suit.any(axis=1)
        flush_poker_hand = np.zeros((self.num_simulations, self.poker_hand_size))
        flush_cards = np.argwhere(flush_suit)
        card_matches_flush= current_hand % 4 == flush_cards[:,1]
        col_of_cards_to_keep = np.argsort(~card_matches_flush, axis=1)[:,:5] # Shape: (num_simulations, n)
        #n_of_a_kind_poker_hand = current_hand[self.row_selector, col_of_cards_to_keep] * n_of_a_kind_mask[:,None]
        return is_flush, flush_poker_hand

    def has_straight(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a straight
        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_straight (np.array): Bool array if each simulation is a straight. Shape: (num_simulations,)
        """
        rank_count_per_straight_ref = rank_count[:, self.straight_references] # Shape: (num_simulations, len(straight_references), poker_hand_size)
        is_straight_per_hand = (rank_count[:, self.straight_references] >= 1).all(axis=2) # Shape: (num_simulations, len(straight_references))
        is_straight = is_straight_per_hand.any(axis=1)
        return is_straight, None

    def has_3_of_a_kind(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a 3-of-a-kind

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_3_of_a_kind (np.array): Bool array if each simulation is a 3-of-a-kind. Shape: (num_simulations,)
        """
        is_3_of_a_kind = self.has_n_of_a_kind(rank_count=rank_count, n=3)
        return is_3_of_a_kind

    def has_two_pair(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has two pairs

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_two_pair (np.array): Bool array if each simulation is a two_pair. Shape: (num_simulations,)
        """
        has_at_least_two_ranks = (rank_count >= 2).sum(axis=1) # Has at least 2 ranks. Shape: (num_simulations,)
        is_two_pair = has_at_least_two_ranks >= 2 # Has at least 2 ranks that have pairs
        return is_two_pair

    def has_2_of_a_kind(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a 2-of-a-kind or a pair

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_2_of_a_kind (np.array): Bool array if each simulation is a 2-of-a-kind. Shape: (num_simulations,)
        """
        is_2_of_a_kind = self.has_n_of_a_kind(rank_count=rank_count, n=2)
        return is_2_of_a_kind

    def has_high_card(self, current_hand, rank_count, suit_count, card_count):
        """
        Checks if array of hands each has a high card
        #TODO: change to always return true? depends on the hierarchy of hands

        Args:
            rank_count (np.array): Array of rank counts. Shape: (num_simulations, NUM_RANKS)
        Returns:
            is_high_card (np.array): Bool array if each simulation is a high card. Shape: (num_simulations,)
        """    
        # at_most_1_count = (rank_count <= 1).all(axis=1)
        # is_not_flush = ~self.has_flush(suit_count=suit_count)
        # is_not_straight = ~self.has_straight(rank_count=rank_count)
        # is_high_card = at_most_1_count & is_not_straight
        is_high_card = np.tile(True, (self.num_simulations,)) # high card is a subset to all poker hands 
        high_card_poker_hand = current_hand[self.row_selector, np.argmax(current_hand % 13, axis=1)][:,None]
        return is_high_card

def main():

    B = Balatro(4, strategy="random", seed=123, verbose=True)
    scores, poker_hands = B.simulate_one_round()

    df = pd.DataFrame({"Poker_hand": poker_hands, "Score": scores})
    df.boxplot(column="Score",by="Poker_hand")
    plt.show()


if __name__ == "__main__":
    main()
