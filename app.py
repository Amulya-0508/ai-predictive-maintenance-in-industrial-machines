import os,time,joblib,numpy as np,pandas as pd,streamlit as st,plotly.graph_objects as go
st.set_page_config(page_title="AI Predictive Maintenance",page_icon="⚙️",layout="wide")
st.title("⚙️ AI-Based Predictive Maintenance for Industrial Machines")
st.caption("Software prototype using simulated industrial sensor telemetry")
MODEL_PATH="predictive_maintenance_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in the repository.")
    st.stop()
model=joblib.load(MODEL_PATH)
if "history" not in st.session_state: st.session_state.history=[]
if "sim_step" not in st.session_state: st.session_state.sim_step=0
FEATURES=["temperature","vibration","current","rpm","operating_hours","machine_age"]
def simulate(machine_id,mode):
    rng=np.random.default_rng()
    if mode=="Normal": vals={"temperature":rng.normal(43,2.8),"vibration":rng.normal(.22,.035),"current":rng.normal(1.15,.10),"rpm":rng.normal(1480,15)}
    elif mode=="Warning": vals={"temperature":rng.normal(59,4),"vibration":rng.normal(.52,.08),"current":rng.normal(1.75,.16),"rpm":rng.normal(1400,28)}
    else: vals={"temperature":rng.normal(75,4.5),"vibration":rng.normal(.90,.10),"current":rng.normal(2.55,.25),"rpm":rng.normal(1280,40)}
    vals["temperature"]=float(np.clip(vals["temperature"],25,100)); vals["vibration"]=float(np.clip(vals["vibration"],.03,1.5)); vals["current"]=float(np.clip(vals["current"],.3,4.5)); vals["rpm"]=int(np.clip(vals["rpm"],900,1600))
    n=int(machine_id.split("-")[-1]); vals["operating_hours"]=800+n*700+st.session_state.sim_step*.5; vals["machine_age"]=1.5+n*1.2
    return vals
def health(r):
    a=np.clip(100-max(r["temperature"]-40,0)*2,0,100); b=np.clip(100-max(r["vibration"]-.2,0)*90,0,100); c=np.clip(100-max(r["current"]-1.1,0)*35,0,100); d=np.clip(100-abs(r["rpm"]-1480)/4,0,100)
    return int(np.clip(.3*a+.3*b+.2*c+.2*d,0,100))
def recommendation(r):
    reasons=[]; actions=[]
    if r["temperature"]>65: reasons.append("high temperature"); actions.append("inspect cooling and operating load")
    if r["vibration"]>.70: reasons.append("high vibration"); actions.append("inspect rotating components, mounting and bearings")
    if r["current"]>2.20: reasons.append("high current"); actions.append("check mechanical load and motor condition")
    if r["rpm"]<1350: reasons.append("reduced RPM"); actions.append("check overload or increased mechanical resistance")
    return "Continue normal monitoring; no immediate maintenance action is indicated." if not reasons else "Detected "+", ".join(reasons)+". Recommended: "+"; ".join(actions)+"."
with st.sidebar:
    st.header("Controls"); machine_id=st.selectbox("Machine",["MOTOR-001","MOTOR-002","MOTOR-003","MOTOR-004"]); mode=st.selectbox("Simulated condition",["Normal","Warning","Critical"]); auto=st.checkbox("Auto refresh every 2 seconds",False)
    if st.button("Generate new reading"): st.session_state.sim_step+=1
r=simulate(machine_id,mode); sample=pd.DataFrame([r])[FEATURES]; prediction=model.predict(sample)[0]; probs=dict(zip(model.classes_,model.predict_proba(sample)[0])); risk=float(probs.get("Critical",0)*100); score=health(r)
st.session_state.history.append({"time":pd.Timestamp.now().strftime("%H:%M:%S"),"temperature":r["temperature"],"vibration":r["vibration"],"current":r["current"],"rpm":r["rpm"],"status":prediction,"risk":risk}); st.session_state.history=st.session_state.history[-60:]
c1,c2,c3,c4,c5=st.columns(5); c1.metric("Temperature",f'{r["temperature"]:.1f} °C'); c2.metric("Vibration",f'{r["vibration"]:.3f} g'); c3.metric("Current",f'{r["current"]:.2f} A'); c4.metric("RPM",str(r["rpm"])); c5.metric("Failure Risk",f"{risk:.1f}%")
lft,rgt=st.columns(2)
with lft: st.subheader(f"{'🟢' if prediction=='Normal' else '🟡' if prediction=='Warning' else '🔴'} Machine Status: {prediction}"); st.progress(score/100); st.write(f"**Machine Health Score:** {score}/100"); st.write(f"**Machine:** {machine_id}")
with rgt:
    st.subheader("AI Class Probabilities")
    for cls in ["Normal","Warning","Critical"]: p=float(probs.get(cls,0)); st.write(f"{cls}: {p*100:.1f}%"); st.progress(p)
st.subheader("Maintenance Recommendation"); msg=recommendation(r)
if prediction=="Critical": st.error(msg)
elif prediction=="Warning": st.warning(msg)
else: st.success(msg)
st.subheader("Live Sensor Trends"); hist=pd.DataFrame(st.session_state.history)
if len(hist)>1:
    for col,title,unit in [("temperature","Temperature","°C"),("vibration","Vibration","g"),("current","Current","A"),("rpm","RPM","RPM")]:
        fig=go.Figure(); fig.add_trace(go.Scatter(x=hist["time"],y=hist[col],mode="lines+markers",name=title)); fig.update_layout(title=title,height=260,margin=dict(l=20,r=20,t=45,b=20),xaxis_title="Time",yaxis_title=unit); st.plotly_chart(fig,use_container_width=True)
else: st.info("Generate a few readings to display live trend graphs.")
with st.expander("Project information"): st.markdown("**AI:** Random Forest Classifier\n\n**Inputs:** Temperature, vibration, current, RPM, operating hours and machine age\n\n**Outputs:** Normal / Warning / Critical, failure-risk estimate, machine health score and maintenance recommendation\n\n**Prototype note:** Sensor readings are simulated and can later be replaced by ESP32/PLC/industrial sensor data.")
if auto: time.sleep(2); st.rerun()
