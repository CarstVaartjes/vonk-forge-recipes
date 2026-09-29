# Inkling SM121 prefill admission cap

On the tested vLLM/SM121 build, four uniform 38-token Inkling prefills hang the
worker in the first model step. C1-C3 complete normally. The engine then dies
when the executor watchdog times out `sample_tokens`; increasing that timeout
does not make the batch complete.

This opt-in scheduler patch limits only the number of **new** prefills admitted
in one scheduler tick. Already-running requests and `max_num_seqs` are
unchanged, allowing six active decode streams. The recipe uses the largest
verified-safe uniform prefill batch, three. C4-C6 must still pass the real
concurrency harness before this workaround is considered validated.
