import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import platform

# 한글 폰트 및 마이너스 기호 깨짐 방지 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 실행 전 지역 및 방역 시나리오 설정 (격자 55x55 고정)
# =====================================================================
GRID_SIZE = 55

print("=" * 60)
print("  SEIQRVD 초고밀도(55x55) 공간 격자 확산 & 지역별 맞춤형 방역 시뮬레이터")
print("=" * 60)
print("[지역 및 환경 특성 선택]")
print("1. 대도시 (인구 밀집형: 높은 감염률, 우수한 의료 인프라/백신 수급)")
print("2. 학교 및 밀집시설 (초고속 전파형: 매우 높은 감염률, 제한된 백신 수급)")
print("3. 지방 및 농어촌 (분산형: 낮은 감염률, 취약한 의료 인프라)")
region_choice = input("선택할 지역 번호를 입력하세요 (1~3, 기본값 1): ").strip()

if region_choice == '2':
    region_name = "학교 및 밀집시설"
    default_beta = 0.42       # 매우 높은 전파력 (밀집 환경)
    BASE_MU = 0.02            # 표준 치사율
    VAX_DAILY_CAP = int(GRID_SIZE * GRID_SIZE * 0.015) # 상대적으로 빡빡한 백신 수급
elif region_choice == '3':
    region_name = "지방 및 농어촌"
    default_beta = 0.22       # 낮은 전파력 (분산 환경)
    BASE_MU = 0.035           # 의료 인프라가 취약하여 약간 높은 치사율
    VAX_DAILY_CAP = int(GRID_SIZE * GRID_SIZE * 0.01)  # 적은 일일 백신 접종 한계
else:
    region_name = "대도시 메가폴리스"
    default_beta = 0.35       # 표준 대도시 감염률
    BASE_MU = 0.025           # 표준 치사율
    VAX_DAILY_CAP = int(GRID_SIZE * GRID_SIZE * 0.025) # 우수한 일일 백신 접종 한계

# 사용자 치사율 및 감염률 커스텀 입력 기능
try:
    beta_input = input(f"감염률(Beta) 입력 [0.1 ~ 0.8] (기본값 {default_beta}): ")
    init_beta = float(beta_input) if beta_input.strip() else default_beta
except:
    init_beta = default_beta

try:
    mu_input = input(f"기본 치사율(Mu) 입력 [0.01 ~ 0.1] (기본값 {BASE_MU}): ")
    BASE_MU = float(mu_input) if mu_input.strip() else BASE_MU
except:
    pass

try:
    v_input = input("백신 접종 시작일 입력 (기본 16일차): ")
    init_vax_day = int(v_input) if v_input.strip() else 16
except:
    init_vax_day = 16

try:
    t_input = input("치료제 개발 완료일 입력 (기본 26일차): ")
    init_tr_day = int(t_input) if t_input.strip() else 26
except:
    init_tr_day = 26

DAYS = 60
SIGMA = 0.28
ALPHA = 0.15
BASE_GAMMA = 0.08

# SEIQRVD 상태 정의
S, E, I, Q, R, V, D = 0, 1, 2, 3, 4, 5, 6

cmap = ListedColormap([
    '#3b82f6',  # S: 미감염자
    '#f59e0b',  # E: 잠복기
    '#ef4444',  # I: 확진자
    '#a855f7',  # Q: 격리자
    '#22c55e',  # R: 회복자
    '#06b6d4',  # V: 백신접종
    '#1e293b'   # D: 사망자
])

# =====================================================================
# 2. 전체 시뮬레이션 미리 계산 (캐싱 구조로 렉 원천 차단)
# =====================================================================
def run_full_simulation(vax_start, tr_day):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    center = GRID_SIZE // 2
    
    # 55x55 고정 격자에서 지역 특성에 따른 초기 감염 분포 세팅
    if region_choice == '2': # 학교/밀집시설 (중앙 집중형 강한 클러스터)
        g[center-3:center+4, center-3:center+4] = S
        g[center-2:center+3, center-2:center+3] = I
    elif region_choice == '3': # 지방/농어촌 (넓게 분산된 소규모 감염원)
        g[:, :] = S
        g[center, center] = I
        g[center-10, center+10] = I
        g[center+10, center-10] = I
    else: # 대도시 메가폴리스 (표준 확산 클러스터)
        g[center-2:center+3, center-2:center+3] = S
        g[center-1:center+2, center-1:center+2] = I
        g[center, center] = I
    
    grid_history = [g.copy()]
    stat_history = {state: [np.sum(g == state)] for state in range(7)}
    
    for day in range(1, DAYS + 1):
        current_grid = grid_history[-1]
        new_grid = current_grid.copy()
        
        gamma = 0.40 if day >= tr_day else BASE_GAMMA
        mu = BASE_MU * 0.15 if day >= tr_day else BASE_MU
        
        if day >= vax_start:
            s_coords = np.argwhere(new_grid == S)
            if len(s_coords) > 0:
                vax_count = min(len(s_coords), VAX_DAILY_CAP)
                chosen_indices = np.random.choice(len(s_coords), vax_count, replace=False)
                for idx in chosen_indices:
                    r_idx, c_idx = s_coords[idx]
                    new_grid[r_idx, c_idx] = V

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                state = current_grid[r, c]
                
                if state == S and new_grid[r, c] == S:
                    r_min, r_max = max(0, r-1), min(GRID_SIZE, r+2)
                    c_min, c_max = max(0, c-1), min(GRID_SIZE, c+2)
                    neighbors = current_grid[r_min:r_max, c_min:c_max]
                    
                    infected_neighbors = np.sum(neighbors == I)
                    exposed_neighbors = np.sum(neighbors == E)
                    
                    infection_prob = 1 - ((1 - init_beta) ** infected_neighbors) * ((1 - init_beta * 0.3) ** exposed_neighbors)
                    if np.random.rand() < infection_prob:
                        new_grid[r, c] = E
                        
                elif state == E:
                    if np.random.rand() < SIGMA:
                        new_grid[r, c] = I
                        
                elif state == I:
                    rand_val = np.random.rand()
                    if rand_val < mu:
                        new_grid[r, c] = D
                    elif rand_val < mu + gamma:
                        new_grid[r, c] = R
                    elif rand_val < mu + gamma + ALPHA:
                        new_grid[r, c] = Q
                        
                elif state == Q:
                    rand_val = np.random.rand()
                    if rand_val < mu * 0.8:
                        new_grid[r, c] = D
                    elif rand_val < mu * 0.8 + gamma:
                        new_grid[r, c] = R
                        
        grid_history.append(new_grid)
        for state in range(7):
            stat_history[state].append(np.sum(new_grid == state))
            
    return grid_history, stat_history

current_vax_day = init_vax_day
current_tr_day = init_tr_day
grid_history, stat_history = run_full_simulation(current_vax_day, current_tr_day)

# =====================================================================
# 3. 화면 레이아웃 구성 (3단 서브플롯)
# =====================================================================
fig = plt.figure(figsize=(13, 10.5))
gs = fig.add_gridspec(4, 2, height_ratios=[4.5, 1.2, 1.2, 0.6], width_ratios=[1, 1.2])

ax_grid = fig.add_subplot(gs[0, 0])
ax_chart = fig.add_subplot(gs[0, 1])
ax_ai = fig.add_subplot(gs[1, :])
ax_overall = fig.add_subplot(gs[2, :])

fig.subplots_adjust(bottom=0.15, top=0.92, hspace=0.45)
fig.suptitle(f'SEIQRVD 초고밀도(55x55) 공간 확산 시뮬레이터 [{region_name}]', fontsize=13, fontweight='bold', y=0.96)

im = ax_grid.imshow(grid_history[0], cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
ax_grid.set_title(f'{region_name} 전파 맵 (55x55)', fontsize=10, fontweight='bold', pad=8)
ax_grid.set_xticks([])
ax_grid.set_yticks([])

legend_labels = [
    plt.Line2D([0], [0], marker='s', color='w', label='S (미감염자)', markerfacecolor='#3b82f6', markersize=6),
    plt.Line2D([0], [0], marker='s', color='w', label='E (잠복기)', markerfacecolor='#f59e0b', markersize=6),
    plt.Line2D([0], [0], marker='s', color='w', label='I (확진자)', markerfacecolor='#ef4444', markersize=6),
    plt.Line2D([0], [0], marker='s', color='w', label='Q (격리자)', markerfacecolor='#a855f7', markersize=6),
    plt.Line2D([0], [0], marker='s', color='w', label='R (회복자)', markerfacecolor='#22c55e', markersize=6),
    plt.Line2D([0], [0], marker='s', color='w', label='V (백신접종)', markerfacecolor='#06b6d4', markersize=6),
    plt.Line2D([0], [0], marker='s', color='w', label='D (사망자)', markerfacecolor='#1e293b', markersize=6),
]
ax_grid.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=7)

ax_chart.set_xlim(0, DAYS)
ax_chart.set_ylim(0, GRID_SIZE * GRID_SIZE)
ax_chart.set_xlabel('경과 일수 (Days)', fontsize=9)
ax_chart.set_ylabel('인구 수 (명)', fontsize=9)
ax_chart.grid(True, alpha=0.3)

lines = {}
colors = {S: '#3b82f6', E: '#f59e0b', I: '#ef4444', Q: '#a855f7', R: '#22c55e', V: '#06b6d4', D: '#1e293b'}
labels = {
    S: 'S: 미감염자', 
    E: 'E: 잠복기', 
    I: 'I: 확진자', 
    Q: 'Q: 격리자', 
    R: 'R: 회복자', 
    V: 'V: 백신접종', 
    D: 'D: 사망자'
}

for state in range(7):
    ls = '--' if state == V else '-'
    lines[state] = ax_chart.plot([], [], label=labels[state], color=colors[state], lw=1.8, linestyle=ls)[0]

vax_line_ref = ax_chart.axvline(x=current_vax_day, color='#06b6d4', linestyle=':', label=f'백신({current_vax_day}일)')
tr_line_ref = ax_chart.axvline(x=current_tr_day, color='#22c55e', linestyle=':', label=f'치료제({current_tr_day}일)')
ax_chart.legend(loc='upper right', fontsize=7, ncol=2)

# 1. 데이별 AI 분석 리포트 박스
ax_ai.axis('off')
ai_box = ax_ai.text(0.5, 0.5, "시뮬레이션 대기 중...", fontsize=9.5, fontweight='bold', 
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', edgecolor='#cbd5e1', lw=1.2))

# 2. 전체 실험 종합 리포트 박스
ax_overall.axis('off')
def get_overall_report_text(vax_d, tr_d):
    total_pop = GRID_SIZE * GRID_SIZE
    max_inf = max(stat_history[I])
    max_inf_day = stat_history[I].index(max_inf)
    final_dead = stat_history[D][-1]
    final_recovered = stat_history[R][-1]
    
    return (
        f"[지역 맞춤형 방역 실험 종합 분석 리포트]\n"
        f"• 선택 지역: {region_name} (고정 격자: 55x55, 총 인구: {total_pop:,}명)\n"
        f"• 맞춤 파라미터: 감염률(β) = {init_beta}, 기본 치사율(μ) = {BASE_MU}, 일일 최대 백신 접종력 = {VAX_DAILY_CAP}명/일\n"
        f"• 방역 골든타임: 백신 접종 시작 (Day {vax_d}) | 치료제 개발 완료 (Day {tr_d})\n"
        f"• 시뮬레이션 결과 요약: 최대 확진자 발생일 (Day {max_inf_day}, peak 확진: {max_inf}명) "
        f"| 최종 회복자: {final_recovered:,}명 ({(final_recovered/total_pop)*100:.1f}%) | 최종 누적 사망자: {final_dead}명"
    )

overall_box = ax_overall.text(0.5, 0.5, get_overall_report_text(current_vax_day, current_tr_day), fontsize=9, fontweight='bold', 
                              ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0f2fe', edgecolor='#7dd3fc', lw=1.2))

# 조작 가이드 텍스트 (최하단)
fig.text(0.5, 0.09, 
         '[조작 가이드]  스페이스바(Space): 일시정지 / 재생  |  ◀ / ▶ 화살표 키: 날짜 수동 탐색 (일시정지 상태에서만 가능)', 
         ha='center', va='center', fontsize=9, fontweight='bold', color='#1e293b',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#e2e8f0', edgecolor='#94a3b8', lw=1))

ax_slider_tr = fig.add_axes([0.15, 0.03, 0.30, 0.02])
ax_slider_vax = fig.add_axes([0.58, 0.03, 0.30, 0.02])
slider_tr = Slider(ax_slider_tr, '치료제 도착일', 15, 50, valinit=current_tr_day, valfmt='%d일')
slider_vax = Slider(ax_slider_vax, '백신 시작일', 5, 45, valinit=current_vax_day, valfmt='%d일')

# =====================================================================
# 4. 초고속 반응형 렌더링 함수
# =====================================================================
current_frame = 0
is_paused = False

x_data_cache = [list(range(i + 1)) for i in range(DAYS + 1)]
y_data_cache = {state: [stat_history[state][:i+1] for i in range(DAYS + 1)] for state in range(7)}

def update_display(frame):
    global current_frame
    current_frame = frame
    
    im.set_array(grid_history[frame])
    
    total_pop = GRID_SIZE * GRID_SIZE
    inf_count = stat_history[I][frame]
    inf_pct = (inf_count / total_pop) * 100
    dead_count = stat_history[D][frame]
    
    status_suffix = " [일시정지됨]" if is_paused else ""
    ax_grid.set_title(f'Day {frame:02d} | 확진자: {inf_count}명 ({inf_pct:.1f}%){status_suffix}', fontsize=10, fontweight='bold', pad=8)
    
    ai_status = f"[Day {frame:02d} AI 방역 분석 리포트]{status_suffix}\n"
    if frame < current_vax_day:
        ai_status += f"• [{region_name}] 초기 확산 단계: 전파 진행 중 (감염 비율: {inf_pct:.1f}%) | 백신 투입 전 방역 대책 필요"
    elif frame < current_tr_day:
        ai_status += f"• [{region_name}] 백신 방어선 가동: 일일 접종(최대 {VAX_DAILY_CAP}명)으로 면역 장벽 형성 중 (현재 확진자: {inf_count}명)"
    else:
        ai_status += f"• [{region_name}] 치료제 투입 및 집단 면역: 확산 진정세 (누적 사망: {dead_count}명) | 방역 안정화"
        
    ai_box.set_text(ai_status)
    
    x_current = x_data_cache[frame]
    for state in range(7):
        lines[state].set_data(x_current, y_data_cache[state][frame])
        
    fig.canvas.draw_idle()

def animate(frame):
    global current_frame
    if not is_paused:
        current_frame = (current_frame + 1) % (DAYS + 1)
        update_display(current_frame)
    return [im] + list(lines.values()) + [ai_box, overall_box]

def on_key(event):
    global is_paused, current_frame
    
    if event.key == ' ':
        is_paused = not is_paused
        update_display(current_frame)
            
    elif is_paused:
        if event.key == 'right':
            current_frame = (current_frame + 1) % (DAYS + 1)
            update_display(current_frame)
        elif event.key == 'left':
            current_frame = (current_frame - 1) % (DAYS + 1)
            update_display(current_frame)

fig.canvas.mpl_connect('key_press_event', on_key)

def update_parameters(val):
    global grid_history, stat_history, current_vax_day, current_tr_day, current_frame, vax_line_ref, tr_line_ref, x_data_cache, y_data_cache
    current_vax_day = int(slider_vax.val)
    current_tr_day = int(slider_tr.val)
    
    grid_history, stat_history = run_full_simulation(current_vax_day, current_tr_day)
    
    x_data_cache = [list(range(i + 1)) for i in range(DAYS + 1)]
    y_data_cache = {state: [stat_history[state][:i+1] for i in range(DAYS + 1)] for state in range(7)}
    
    overall_box.set_text(get_overall_report_text(current_vax_day, current_tr_day))
    
    vax_line_ref.remove()
    tr_line_ref.remove()
    vax_line_ref = ax_chart.axvline(x=current_vax_day, color='#06b6d4', linestyle=':', label=f'백신({current_vax_day}일)')
    tr_line_ref = ax_chart.axvline(x=current_tr_day, color='#22c55e', linestyle=':', label=f'치료제({current_tr_day}일)')
    ax_chart.legend(loc='upper right', fontsize=7, ncol=2)
    
    current_frame = min(current_frame, DAYS)
    update_display(current_frame)

slider_vax.on_changed(update_parameters)
slider_tr.on_changed(update_parameters)

ani = animation.FuncAnimation(fig, animate, frames=DAYS+1, interval=80, blit=False)

plt.show()
