"""Frequent English function/content words, used only to score whether
extracted text looks like real prose (see extract.text_quality)."""

COMMON_WORDS = frozenset(
    """
    a about above across act action after again against ago air all
    almost along already also although always am among an and another any
    anything are area areas around as asked at available away back be
    became because become been before began behind being best better between big
    board body both boy brought business but by called came can cannot
    car case certain change children church city close college come community company
    could country course court day days death development did didn different do
    does don done door down during each early economic either end enough
    even ever every example experience eyes face fact family far federal feet
    felt few field find first five for force form found four free
    from full gave general get give given go god going good got
    government great group had half hand hands has have having he head
    heard held help her here high him himself his history home house
    how however human i if important in individual information interest into is
    it its itself job john just keep kind knew know known large
    last later law least left less let life light like line little
    local long look looking love made major make making man many matter
    may me means members men might mind miss moment money more most
    mr mrs much must my name national need never new next night
    no not nothing now number of off office often old on once
    one only open or order other others our out over own part
    past people per perhaps period place point political position possible power present
    president problem problems program public put question quite rather real really reason
    result right room said same saw say school second see seem seemed
    seems seen sense service set several shall she should show side since
    small so social society some something south special state still street study
    such sure system take taken tell than that the their them themselves
    then there these they thing things think this those though thought three
    through thus time times to today together told too took toward true
    turn turned two under united until up upon us use used very
    want war was water way we week well went were west what
    when where whether which while white who whole whose why will with
    within without word words work world would year years yet york you
    young your
    """.split()
)
