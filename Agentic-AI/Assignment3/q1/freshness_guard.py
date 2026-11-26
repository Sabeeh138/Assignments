"""Provenance guard for real pollutant responses, independent of any LLM."""
from datetime import datetime,timezone
import math

def observed_status(values,now=None,max_age_hours=48):
    now=now or datetime.now(timezone.utc)
    issues=[];ages={}
    for key in ['pm25','pm10']:
        item=values.get(key)
        if not isinstance(item,dict):issues.append(key+': missing');continue
        value=item.get('value')
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            issues.append(key+': invalid value');continue
        try:
            age=(now-datetime.fromisoformat(item['datetime']['utc'].replace('Z','+00:00'))).total_seconds()/3600
            ages[key]=age
            if age<0 or age>max_age_hours:issues.append(key+': stale or future timestamp')
        except (KeyError,ValueError,TypeError):issues.append(key+': invalid timestamp')
    return {'usable_for_current_summary':not issues,'age_hours':ages,'issues':issues,
      'action':'Use readings with original source and observation time' if not issues else 'Withhold current pollution number; state data unavailable/stale; retain provenance for audit'}
