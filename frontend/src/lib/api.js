const BASE=import.meta.env.VITE_API_BASE_URL||"http://localhost:8000/api";
export const api={get:(path)=>fetch(BASE+path).then(r=>{if(!r.ok)throw new Error("Request failed");return r.json()}),post:(path,body)=>fetch(BASE+path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json())};
