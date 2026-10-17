# Marhba Evaluation Questions

Test set for evaluating retrieval and generation quality. Each question lists the expected source and the key fact.

## Customs & Duty-Free

1. **What is the duty-free allowance for personal items in Algeria?**
   - Expected source: `douane-2024-traveler-leaflet`
   - Key fact: 150,000 DA

2. **How many cigarettes can I bring into Algeria duty-free?**
   - Expected source: `douane-2024-traveler-leaflet`
   - Key fact: 200 cigarettes, or 50 cigars, or 250g combination

3. **How much wine can I bring into Algeria?**
   - Expected source: `douane-2024-traveler-leaflet`
   - Key fact: 2 liters of wine or 1 liter of spirits

## Currency

4. **How much cash must I declare when entering Algeria?**
   - Expected source: `douane-currency-declaration-2026`
   - Key fact: over 1,000 euros (or equivalent)
   - Conflict: UK FCDO says 5,000 EUR — correct answer should prefer 1,000 EUR

5. **How much foreign currency can I take out of Algeria?**
   - Expected source: `douane-currency-declaration-2026`
   - Key fact: 7,500 euros per calendar year

6. **How many Algerian dinars can I bring in or take out?**
   - Expected source: `douane-currency-declaration-2026`
   - Key fact: 10,000 DA, resident travelers only

## Visa & Entry

7. **Do I need a visa to visit Algeria?**
   - Expected source: `mfa-entry-visa-2026`
   - Key fact: yes, for most foreign nationals (exceptions listed in London Consulate source)

8. **How long must my passport be valid to enter Algeria?**
   - Expected source: `uk-fcdo-entry-req-2026`
   - Key fact: 6 months beyond arrival

9. **What happens if I overstay my visa in Algeria?**
   - Expected source: `uk-fcdo-entry-req-2026`
   - Key fact: up to 3 months in prison, criminal prosecution

10. **Which nationalities are visa-exempt for Algeria?**
    - Expected source: `dz-consulate-london-visa-2026`
    - Key fact: Libya, Malaysia, Maldives, Mali, Mauritania, Morocco (stale), Sahrawi Arab Democratic Republic, Seychelles, Tunisia

## Prohibited Items & Equipment

11. **Can I bring a drone to Algeria?**
    - Expected source: `douane-2024-traveler-leaflet`
    - Key fact: no, drones are sensitive equipment requiring specific regulations

12. **Can I bring binoculars to Algeria?**
    - Expected source: `douane-2024-traveler-leaflet`
    - Key fact: no, binoculars are sensitive equipment

## Laws & Behavior

13. **Can I drink alcohol in public in Algeria?**
    - Expected source: `uk-fcdo-safety-2026`
    - Key fact: no, illegal to drink or be under the influence in public

14. **Is it legal to photograph government buildings in Algeria?**
    - Expected source: `uk-fcdo-safety-2026`
    - Key fact: certain photography equipment requires authorization; drones/binoculars/telescopes prohibited without authorization

15. **Can I bring my pet dog to Algeria?**
    - Expected source: `douane-2024-traveler-leaflet`
    - Key fact: yes, with health certificate (<8 days) and rabies vaccination record

## Edge Cases / Refusals

16. **Can I bring my prescription medication to Algeria?**
    - Expected behavior: system says "I don't have official information on this"
    - (Gap: no official Algerian traveler medication source in corpus)

17. **Can I bring my pet parrot to Algeria?**
    - Expected behavior: system says "I don't have official information on this"
    - (Gap: pets section covers dogs/cats/hamsters, not birds specifically)

## Evaluation Rubric

For each question, record:

| Column | Meaning |
|---|---|
| `retrieved_source` | Which `doc_id` appears in top-3 chunks |
| `retrieval_correct` | Does the source match the expected source? (Y/N) |
| `answer_correct` | Does the answer contain the key fact? (Y/N) |
| `cited` | Does the answer cite the source? (Y/N) |
| `notes` | Any issues |