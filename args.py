#!/usr/bin/env python3

import argparse
import sys

class Args:
    def __init__(self, name):
        self.name = name

    def set_args(self):
        parser = self.set_parser(self.name)
        parser = self.original_arguments(parser)
        parser = self.added_arguments(parser)
        args = self.prep_args(parser)
        return args

    def set_parser(self, name):
        parser = argparse.ArgumentParser(name)
        return parser

    def original_arguments(self, parser):
        raise NotImplementedError

    def added_arguments(self, parse):
        raise NotImplementedError

    def prep_args(self, parser):
        try:
            is_running_test = [True if 'pytest' in i else False for i in sys.argv]
            if any(is_running_test):
                args = parser.parse_args([])
            else:
                args = parser.parse_args()
        except:
            parser.print_help()
            sys.exit(0)
        return args
