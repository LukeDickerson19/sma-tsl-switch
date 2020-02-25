import re, time

# this script prints overtop the previous text (over multiple lines)


class BlockPrinter:
    def __init__(self):
        self.num_lines = 0
        self.max_line_width = 0

    def clear(self):
        for _ in range(self.num_lines):
            print('\x1b[A' + '\r' + ' '*self.max_line_width + '\r', end='')

    def print(self, text, end='\n'):

        # clear previous text by overwriting non-spaces with spaces
        self.clear()

        # Print new text
        text += end
        print(text, end='')
        self.num_lines = text.count('\n')
        self.max_line_width = max(map(
            lambda line : len(line), text.split('\n')
        ))




def test():

    bp = BlockPrinter()
    SLEEP_TIME = 0.5 # seconds

    # new text has shorter lines
    bp.print('aaaa')
    time.sleep(SLEEP_TIME)
    bp.print('bbb')
    time.sleep(SLEEP_TIME)
    bp.print('cc')
    time.sleep(SLEEP_TIME)
    bp.print('d')
    time.sleep(SLEEP_TIME)
    bp.print('')
    time.sleep(SLEEP_TIME)

    # new text has longer lines
    bp.print('a')
    time.sleep(SLEEP_TIME)
    bp.print('bb')
    time.sleep(SLEEP_TIME)
    bp.print('ccc')
    time.sleep(SLEEP_TIME)
    bp.print('dddd')
    time.sleep(SLEEP_TIME)
    bp.print('')
    time.sleep(SLEEP_TIME)

    # new text has more lines
    bp.print('a')
    time.sleep(SLEEP_TIME)
    bp.print('b\nb')
    time.sleep(SLEEP_TIME)
    bp.print('c\nc\nc')
    time.sleep(SLEEP_TIME)
    bp.print('d\nd\nd\nd')
    time.sleep(SLEEP_TIME)
    bp.print('')
    time.sleep(SLEEP_TIME)

    # new text has less lines
    bp.print('a\na\na\na')
    time.sleep(SLEEP_TIME)
    bp.print('b\nb\nb')
    time.sleep(SLEEP_TIME)
    bp.print('c\nc')
    time.sleep(SLEEP_TIME)
    bp.print('d')
    time.sleep(SLEEP_TIME)
    bp.print('')
    time.sleep(SLEEP_TIME)

if __name__ == '__main__':
    test()
