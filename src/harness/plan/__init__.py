"""plan: symptom -> subsystem -> minimal targetted collector set.

Prevents collecting everything up front. Given initial symptoms/error counters, it
classifies the likely subsystem and returns the smallest ordered collector set.
"""