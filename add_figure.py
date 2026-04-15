import json

with open('results/human_eval_packet.json', encoding='utf-8') as f:
    packet = json.load(f)

# Trapezium matching textbook:
# AB on TOP (A left, B right), DC on BOTTOM (D left, C right)
# F outside top-left where CE produced and BA produced meet
# E on AD (midpoint)
trapezium_svg = '''<svg viewBox="0 0 320 210" xmlns="http://www.w3.org/2000/svg" style="max-width:300px;display:block;margin:auto">

  <!-- Trapezium ABCD: AB top || DC bottom -->
  <polygon points="130,45 265,45 285,170 40,170"
           fill="#EEF2FF" stroke="#4472C4" stroke-width="2"/>

  <!-- Labels -->
  <text x="125" y="35"  font-size="14" font-weight="bold" fill="#2E4057">A</text>
  <text x="268" y="35"  font-size="14" font-weight="bold" fill="#2E4057">B</text>
  <text x="288" y="185" font-size="14" font-weight="bold" fill="#2E4057">C</text>
  <text x="25"  y="185" font-size="14" font-weight="bold" fill="#2E4057">D</text>

  <!-- E on AD midpoint: A(130,45) D(40,170) => (85,108) -->
  <circle cx="85" cy="108" r="4" fill="#E24B4A"/>
  <text x="92" y="106" font-size="13" font-weight="bold" fill="#E24B4A">E</text>

  <!-- F top-left: BA produced beyond A, CE produced meets there -->
  <text x="6" y="42" font-size="14" font-weight="bold" fill="#E24B4A">F</text>

  <!-- BA produced: B(265,45) -> A(130,45) -> F(18,45) -->
  <line x1="265" y1="45" x2="18" y2="45" stroke="#4472C4" stroke-width="2"/>

  <!-- CE produced through E to F: C(285,170)->E(85,108)->F(18,45) -->
  <line x1="285" y1="170" x2="18" y2="45"
        stroke="#E24B4A" stroke-width="1.5" stroke-dasharray="5,3"/>

  <!-- FD dashed: F(18,45) to D(40,170) -->
  <line x1="18" y1="45" x2="40" y2="170"
        stroke="#555" stroke-width="1.2" stroke-dasharray="4,3"/>

  <!-- AC dashed: A(130,45) to C(285,170) -->
  <line x1="130" y1="45" x2="285" y2="170"
        stroke="#555" stroke-width="1.2" stroke-dasharray="4,3"/>

</svg>'''

# Isosceles trapezoid: AB || DC, AD = BC
trapezoid_svg = '''<svg viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg" style="max-width:280px;display:block;margin:auto">

  <polygon points="40,145 260,145 210,40 90,40"
           fill="#EEF2FF" stroke="#4472C4" stroke-width="2"/>

  <text x="27"  y="163" font-size="14" font-weight="bold" fill="#2E4057">A</text>
  <text x="263" y="163" font-size="14" font-weight="bold" fill="#2E4057">B</text>
  <text x="213" y="34"  font-size="14" font-weight="bold" fill="#2E4057">C</text>
  <text x="78"  y="34"  font-size="14" font-weight="bold" fill="#2E4057">D</text>

  <text x="125" y="168" font-size="11" fill="#555">AB = 8 cm</text>
  <text x="128" y="30"  font-size="11" fill="#555">DC = 5 cm</text>

  <line x1="150" y1="145" x2="150" y2="40"
        stroke="#888" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="154" y="100" font-size="11" fill="#555">h = 4 cm</text>

</svg>'''

for item in packet:
    if item['test_id'] == 93:
        item['figure_svg'] = trapezium_svg
    elif item['test_id'] == 61:
        item['figure_svg'] = trapezoid_svg
    else:
        item['figure_svg'] = None

with open('results/human_eval_packet.json', 'w', encoding='utf-8') as f:
    json.dump(packet, f, ensure_ascii=False, indent=2)

print("Done! Figure updated to match textbook.")
