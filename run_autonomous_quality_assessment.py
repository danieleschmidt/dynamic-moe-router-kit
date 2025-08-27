#!/usr/bin/env python3
"""Run autonomous quality assessment without interactive prompts."""

import sys
sys.path.append('.')

from autonomous_sdlc_quality_integration import AutonomousSDLCQualitySystem

def main():
    """Run autonomous quality assessment and save results."""
    print("🚀 AUTONOMOUS SDLC QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Initialize quality system
    print("⚡ Initializing integrated quality system...")
    quality_system = AutonomousSDLCQualitySystem()
    
    # Run comprehensive assessment
    print("🔍 Running comprehensive quality assessment...")
    report = quality_system.run_comprehensive_quality_assessment()
    
    # Display results
    print(f"\n📊 ASSESSMENT RESULTS")
    print(f"Overall Status: {report.overall_status}")
    print(f"Overall Quality Score: {report.combined_metrics.get('overall_quality_score', 0):.1f}/100")
    print(f"Production Ready: {'✅' if report.production_readiness else '❌'}")
    print(f"Research Validated: {'✅' if report.research_validation else '❌'}")
    
    # Show combined metrics
    print(f"\n🎯 COMBINED METRICS:")
    for metric, value in report.combined_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.1f}")
    
    # Show top recommendations
    if report.recommendations:
        print(f"\n📋 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    # Show autonomous actions
    if report.autonomous_actions_taken:
        print(f"\n🤖 AUTONOMOUS ACTIONS TAKEN:")
        for action in report.autonomous_actions_taken:
            print(f"  ✅ {action}")
    else:
        print(f"\n🤖 No autonomous actions taken (autonomous engine not available)")
    
    # Export reports
    print(f"\n📄 EXPORTING REPORTS...")
    json_report = quality_system.export_quality_report(report, "json")
    md_report = quality_system.export_quality_report(report, "markdown")
    print(f"  JSON Report: {json_report}")
    print(f"  Markdown Report: {md_report}")
    
    # Generate dashboard
    print(f"\n📈 QUALITY DASHBOARD:")
    dashboard = quality_system.generate_quality_dashboard()
    print(f"  System Status: {dashboard['system_status']}")
    capabilities = dashboard['capabilities']
    active_capabilities = sum(capabilities.values())
    total_capabilities = len(capabilities)
    print(f"  Active Capabilities: {active_capabilities}/{total_capabilities}")
    
    for capability, active in capabilities.items():
        status = "✅" if active else "❌"
        print(f"    {status} {capability.replace('_', ' ').title()}")
    
    print(f"\n🎉 Autonomous SDLC Quality Assessment complete!")
    print(f"📊 Overall Quality Score: {report.combined_metrics.get('overall_quality_score', 0):.1f}/100")
    print(f"🚀 Production Ready: {report.production_readiness}")
    print(f"🔬 Research Validated: {report.research_validation}")
    
    return report

if __name__ == "__main__":
    main()